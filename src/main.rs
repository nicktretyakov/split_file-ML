use tokio::io::{AsyncReadExt, BufReader};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Simple text segmentation tool");
    println!("Reading from stdin...");
    
    let mut buffer = String::new();
    let mut accumulated_text = String::new();
    let mut sentence_count = 0;
    let mut stream = BufReader::new(tokio::io::stdin());
    let mut buf = [0u8; 1024];

    loop {
        // Обработка всех полных предложений в буфере
        while let Some(sentence) = extract_sentence(&mut buffer) {
            // Простая эвристика на основе длины предложения и ключевых слов
            let should_start_new_segment = should_segment(&accumulated_text, &sentence, sentence_count);
            
            if sentence_count == 0 || should_start_new_segment {
                if !accumulated_text.is_empty() {
                    // Вывод предыдущего сегмента
                    println!("----- ЧАСТЬ -----\n{}", accumulated_text);
                }
                // Начало нового сегмента
                accumulated_text = sentence;
                sentence_count = 1;
            } else {
                // Добавление предложения к текущему сегменту
                accumulated_text.push(' ');
                accumulated_text.push_str(&sentence);
                sentence_count += 1;
            }
        }
        
        // Чтение следующего куска данных из потока
        let n = stream.read(&mut buf).await?;
        if n == 0 {
            // Вывод последнего сегмента
            if !accumulated_text.is_empty() {
                println!("----- ЧАСТЬ -----\n{}", accumulated_text);
            }
            break;
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

// Простая эвристика для определения смены сегмента
fn should_segment(current_text: &str, new_sentence: &str, sentence_count: usize) -> bool {
    // Начинаем новый сегмент каждые 5 предложений
    if sentence_count >= 5 {
        return true;
    }
    
    // Ключевые слова, указывающие на смену темы
    let topic_markers = [
        "однако", "тем не менее", "в то же время", "с другой стороны",
        "кроме того", "более того", "в дополнение", "также",
        "например", "в частности", "а именно",
        "в заключение", "таким образом", "итак", "следовательно"
    ];
    
    let sentence_lower = new_sentence.to_lowercase();
    for marker in &topic_markers {
        if sentence_lower.contains(marker) {
            return true;
        }
    }
    
    // Если предложение значительно длиннее или короче предыдущих
    let avg_length = if current_text.is_empty() { 0 } else { current_text.len() / sentence_count.max(1) };
    let new_length = new_sentence.len();
    
    if avg_length > 0 && (new_length > avg_length * 2 || new_length < avg_length / 2) {
        return sentence_count > 2; // Но не слишком рано
    }
    
    false
}
