use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModel};
use ndarray::Array1;
use tokio::io::{AsyncReadExt, BufReader};
use std::sync::Arc;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Инициализация модели для вычисления вложений предложений
    let model = Arc::new(SentenceEmbeddingsBuilder::new("bert-base-nli-mean-tokens").create_model()?);
    let threshold = 0.5; // Порог для определения смены темы
    let mut buffer = String::new();
    let mut accumulated_text = String::new();
    let mut centroid: Option<Array1<f32>> = None;
    let mut count = 0;
    let mut stream = BufReader::new(tokio::io::stdin());
    let mut buf = [0u8; 1024];

    loop {
        // Обработка всех полных предложений в буфере
        while let Some(sentence) = extract_sentence(&mut buffer) {
            let model = model.clone();
            // Вычисление вложения предложения в отдельном потоке
            let embedding = tokio::task::spawn_blocking(move || {
                let embeddings = model.encode(&[&sentence])?;
                Ok(embeddings[0].clone())
            }).await??;
            // Нормализация вложения
            let embedding = embedding / embedding.dot(&embedding).sqrt();

            if count == 0 {
                // Первое предложение в сегменте
                accumulated_text = sentence;
                centroid = Some(embedding.clone());
                count = 1;
            } else {
                // Вычисление сходства с текущим центроидом
                let current_centroid = centroid.as_ref().unwrap();
                let similarity = current_centroid.dot(&embedding) / current_centroid.dot(current_centroid).sqrt();
                if similarity >= threshold {
                    // Добавление предложения к текущему сегменту
                    accumulated_text.push_str(&sentence);
                    *centroid.as_mut().unwrap() += &embedding;
                    count += 1;
                } else {
                    // Вывод сегмента при смене темы
                    println!("----- ЧАСТЬ -----\n{}", accumulated_text);
                    accumulated_text = sentence;
                    centroid = Some(embedding.clone());
                    count = 1;
                }
            }
        }
        // Чтение следующего куска данных из потока
        let n = stream.read(&mut buf).await?;
        if n == 0 {
            break; // Для конечных потоков, хотя предполагается бесконечность
        }
        buffer.push_str(&String::from_utf8_lossy(&buf[..n]));
    }
    Ok(())
}

// Функция для извлечения предложений из буфера
fn extract_sentence(buffer: &mut String) -> Option<String> {
    for (i, c) in buffer.char_indices() {
        if c == '.' || c == '?' || c == '!' {
            if i + 1 < buffer.len() && buffer.as_bytes()[i + 1] == b' ' {
                let sentence = buffer[..=i].to_string();
                *buffer = buffer[i + 2..].to_string();
                return Some(sentence);
            }
        }
    }
    None
}
