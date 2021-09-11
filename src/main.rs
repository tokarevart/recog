use tokio_postgres::{NoTls, Client};
use tokio::sync::Mutex;
use std::sync::Arc;

fn get_by_isize(coll: &Vec<String>, idx: isize) -> Option<&String> {
    if idx < 0 {
        None
    } else {
        coll.get(idx as usize)
    }
}

fn nbhood_by_iter(coll: &Vec<String>, idx: usize, nbs_idces: impl Iterator<Item=isize>) -> Vec<(String, i32)> {
    nbs_idces.map(|x| (get_by_isize(coll, idx as isize + x), -x as i32))
             .filter(|x| x.0.is_some())
             .map(|x| (x.0.unwrap().clone(), x.1))
             .collect()
}

const RANGE: isize = 3;

fn left_nbhood(coll: &Vec<String>, idx: usize) -> Vec<(String, i32)> {
    nbhood_by_iter(coll, idx, -RANGE..0)
}

fn right_nbhood(coll: &Vec<String>, idx: usize) -> Vec<(String, i32)> {
    nbhood_by_iter(coll, idx, 1..=RANGE)
}

async fn upsert_from_file(client: Arc<Mutex<Client>>, filename: &str) {
    let sentences = std::fs::read_to_string(filename)
        .unwrap()
        .replace("\n", " ")
        .to_ascii_lowercase()
        .split(|c| c == '.' || c == '(' || c == ')' || c == ';' || c == ':')
        .filter(|x| !x.is_empty())
        .map(|x| x.trim().to_owned())
        .collect::<Vec<String>>();

    let mut handles = Vec::with_capacity(sentences.len());
    for words in sentences.into_iter()
                          .map(|x| x.split_ascii_whitespace()
                                    .map(|x| x.to_owned()) 
                          .collect::<Vec<String>>()) {
        let client = client.clone();
        handles.push(tokio::spawn(async move {
            for (cword_idx, cword) in words.iter().enumerate() {
                for (nb, dist) in left_nbhood(&words, cword_idx) {
                    client.lock().await.execute("call upsert_pair($1::text, $2::text, $3::integer)", &[&nb, &cword, &dist]).await.unwrap();
                }
                for (nb, dist) in right_nbhood(&words, cword_idx) {
                    client.lock().await.execute("call upsert_pair($1::text, $2::text, $3::integer)", &[&cword, &nb, &dist]).await.unwrap();
                }
            }
        }));
    }

    for h in handles {
        h.await.unwrap();
    }
}

async fn recog_file(client: &Mutex<Client>, filename: &str) -> String {
    recog_string(client, &std::fs::read_to_string(filename).unwrap()).await
}

async fn recog_string(client: &Mutex<Client>, sentence: &str) -> String {
    let words = sentence
        .to_ascii_lowercase()
        .split_ascii_whitespace()
        .map(|x| x.to_owned())
        .collect::<Vec<String>>();

    let mut recoged_words = Vec::with_capacity(words.len());
    for (i, word) in words.iter().enumerate() {
        if !word.contains(|c| c == '_' || c == '%') {
            recoged_words.push(word.clone());
            continue;
        }

        let lnbh = left_nbhood(&words, i);
        let rnbh = right_nbhood(&words, i);
        let (lwords, ldists): (Vec<String>, Vec<i32>) = lnbh.into_iter().unzip();
        let (rwords, rdists): (Vec<String>, Vec<i32>) = rnbh.into_iter().unzip();

        recoged_words.push(
            client.lock().await.query(
                "select min_inconsistency_word($1::text[], $2::integer[], $3::text, $4::text[], $5::integer[])", 
                &[&lwords, &ldists, word, &rwords, &rdists]
            ).await.unwrap()[0].get(0)
        );
    }
    
    recoged_words.join(&" ")
}

#[tokio::main] 
async fn main() {
    let (client, connection) =
        tokio_postgres::connect("host=localhost user=postgres password=123", NoTls).await.unwrap();
    let client = Arc::new(Mutex::new(client));

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("connection error: {}", e);
        }
    });

    // upsert_from_file(client.clone(), "train.txt").await;
    println!("{}", recog_string(&client, "based on NVIDIA % is % to simplify the").await);
}