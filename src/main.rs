use ndarray::{Array1, Array2};
use regex::Regex;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use std::collections::{HashMap, HashSet};

fn tokenize(sentence: &str) -> Vec<String> {
    sentence
        .split_whitespace()
        .map(|s| s.to_lowercase())
        .collect()
}

struct TfidfModel {
    vocab: Vec<String>,
    idf: HashMap<String, f64>,
}

impl TfidfModel {
    fn new(sentences: &[&str]) -> Self {
        let mut vocab_set = HashSet::new();
        for sentence in sentences {
            for word in tokenize(sentence) {
                vocab_set.insert(word);
            }
        }
        let vocab: Vec<String> = vocab_set.into_iter().collect();
        let mut idf = HashMap::new();
        let n_docs = sentences.len() as f64;
        for word in &vocab {
            let df = sentences
                .iter()
                .filter(|s| tokenize(s).contains(word))
                .count() as f64;
            let idf_value = (n_docs / (df + 1.0)).ln();
            idf.insert(word.clone(), idf_value);
        }
        TfidfModel { vocab, idf }
    }

    fn transform(&self, sentence: &str) -> Array1<f64> {
        let words = tokenize(sentence);
        let mut tf = HashMap::new();
        for word in &words {
            *tf.entry(word.clone()).or_insert(0.0) += 1.0;
        }
        let mut vector = Array1::zeros(self.vocab.len());
        for (i, word) in self.vocab.iter().enumerate() {
            if let Some(count) = tf.get(word) {
                let tf_value = *count / words.len() as f64;
                let idf_value = self.idf[word];
                vector[i] = tf_value * idf_value;
            }
        }
        vector
    }
}

fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot_product = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

fn extract_features(sentences: &[&str]) -> Array2<f64> {
    let tfidf_model = TfidfModel::new(sentences);
    let tfidf_vectors: Vec<Array1<f64>> =
        sentences.iter().map(|s| tfidf_model.transform(s)).collect();

    let re = Regex::new(r"(?i)Introduction|Methods|Results").unwrap();

    let mut features = Vec::new();
    for i in 0..sentences.len() {
        let has_section_keyword = if re.is_match(sentences[i]) { 1.0 } else { 0.0 };
        let cosine_dist = if i > 0 {
            let a = &tfidf_vectors[i];
            let b = &tfidf_vectors[i - 1];
            1.0 - cosine_similarity(a, b)
        } else {
            0.0
        };
        let sentence_length = tokenize(sentences[i]).len() as f64;
        features.push(vec![has_section_keyword, cosine_dist, sentence_length]);
    }
    Array2::from_shape_vec(
        (sentences.len(), 3),
        features.into_iter().flatten().collect(),
    )
    .unwrap()
}

fn predict_segments(model: &RandomForestClassifier<f64>, new_sentences: &[&str]) -> Vec<f64> {
    let X_new = extract_features(new_sentences);
    let X_new_vec: Vec<Vec<f64>> = X_new.outer_iter().map(|row| row.to_vec()).collect();
    let X_new_dm = DenseMatrix::from_2d_vec(&X_new_vec);
    model.predict(&X_new_dm).unwrap()
}

fn main() {
    let texts = vec![
        "Introduction. This study explores...",
        "We used a novel method...",
        "Methods. The experiment was...",
        "Data was collected from...",
        "Results. The findings show...",
    ];
    let labels = vec![1.0, 0.0, 1.0, 0.0, 1.0];

    let X = extract_features(&texts);
    let X_vec: Vec<Vec<f64>> = X.outer_iter().map(|row| row.to_vec()).collect();
    let X_train = DenseMatrix::from_2d_vec(&X_vec);

    let model = RandomForestClassifier::<f64>::fit(&X_train, &labels, Default::default()).unwrap();

    let new_texts = vec![
        "Abstract. This paper discusses...",
        "The approach was tested...",
        "Conclusion. We found that...",
    ];

    let predictions = predict_segments(&model, &new_texts);

    for (text, pred) in new_texts.iter().zip(predictions.iter()) {
        println!(
            "Sentence: {} | New segment: {}",
            text,
            if *pred == 1.0 { "true" } else { "false" }
        );
    }
}
