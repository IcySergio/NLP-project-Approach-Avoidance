import os
import re
import string
from collections import Counter

import docx
import fitz
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# --- 0. Download NLTK resources ---

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- 1. Defining keywords and parameters ---
APPROACH_KEYWORDS = {
    'achieve', 'achievement', 'advance', 'advantage', 'aspire', 'benefit',
    'best', 'build', 'challenge', 'contribute', 'create', 'curious', 'desire',
    'develop', 'eager', 'effective', 'efficient', 'encourage', 'enjoy',
    'enthusiasm', 'excited', 'explore', 'fascinating', 'fast', 'forward',
    'gain', 'goal', 'good', 'great', 'grow', 'growth', 'help', 'hope',
    'idea', 'improve', 'improvement', 'increase', 'initiative', 'innovate',
    'innovation', 'interest', 'learn', 'learning', 'love', 'master',
    'motivation', 'motivated', 'new', 'opportunity', 'optimize', 'passion',
    'positive', 'potential', 'progress', 'promote', 'pursue', 'ready',
    'satisfaction', 'satisfied', 'satisfy', 'seek', 'skill', 'solution',
    'start', 'strength', 'succeed', 'success', 'support', 'towards', 'train',
    'try', 'understand', 'value', 'want', 'welcome', 'win', 'wish'
}

AVOIDANCE_KEYWORDS = {
    'afraid', 'anxiety', 'avoid', 'bad', 'blame', 'block', 'burden',
    'burnout', 'complain', 'concern', 'conflict', 'confused', 'constraint',
    'criticism', 'danger', 'defend', 'delay', 'deny', 'difficult',
    'difficulty', 'disadvantage', 'disappointed', 'dislike',
    'dissatisfaction', 'dissatisfied', 'dissatisfy', 'doubt', 'drag',
    'dread', 'escape', 'fail', 'failure', 'fear', 'fine', 'force',
    'frustrated', 'hard', 'hardship', 'hate', 'hesitate', 'hurdle', 'ignore',
    'impossible', 'inadequate', 'inhibit', 'insecure', 'issue', 'lack',
    'limit', 'limitation', 'lose', 'loss', 'mandatory', 'minimize',
    'mistake', 'must', 'need', 'negative', 'neglect', 'nervous', 'never', 'oblige',
    'obligation', 'obstacle', 'pain', 'panic', 'penalty', 'prevent',
    'problem', 'protect', 'quit', 'reduce', 'reject',
    'reluctant', 'resist', 'risk', 'risky', 'stagnant', 'stop', 'stress',
    'struggle', 'stuck', 'suffer', 'threat', 'tired', 'trouble', 'unable',
    'uncertain', 'uncomfortable', 'unhappy', 'unpleasant', 'unsafe',
    'unsure', 'urgent', 'weakness', 'without', 'worry', 'worried', 'wrong'
}

NEGATION_WORDS = {
    'not', 'never', 'no', 'without', "n't", 'hardly', 'rarely', 'seldom'
}
NEGATION_WINDOW = 5

# --- 2. Initializing NLP Tools ---
lemmatizer = WordNetLemmatizer()
stop_words_english = set(stopwords.words('english'))
stop_words_english -= NEGATION_WORDS
ADDITIONAL_STOP_WORDS = {
    'speaker', 'yeah', 'okay', 'know', 'think', 'really', 'say', 'get',
    'right', 'lot', 'unknown', 'yes', 'make', 'kind', 'well', 'maybe',
    'much', 'also', 'even', 'actually', 'mm-hmm', 'affirmative',
    'interviewee', 'interviewer', 'guest', 'uh', 'um', 'gonna', 'wanna',
    'like'
}
stop_words_english.update(ADDITIONAL_STOP_WORDS)
punctuation_set = set(string.punctuation)


# --- 3. File Reading Functions ---
def read_text_file(filepath: str) -> str | None:
    """
    Reads text content from a .txt file.

    Args:
        filepath: Path to the file.

    Returns:
        The file contents as a string, or None if an error occurs.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading text file {filepath}: {e}")
        return None


def read_docx_file(filepath: str) -> str | None:
    """
    Reads text content from a .docx file.

    Args:
        filepath: Path to the file.

    Returns:
        The combined text from the document’s paragraphs, or None if an error occurs.
    """
    try:
        doc = docx.Document(filepath)
        return '\n'.join([
            para.text for para in doc.paragraphs if para.text.strip()
        ])
    except Exception as e:
        print(f"Error reading .docx file {filepath}: {e}")
        return None


def read_tsv_file(filepath: str,
                  text_column_indices: list[int] = None
                  ) -> str | None:
    """
    Reads text content from a .tsv file.

    Args:
        filepath: Path to the file.
        text_column_indices: List of column indices that contain text
                             (defaults to [0]).

    Returns:
        The combined text from the specified columns, or None if an error occurs.
    """
    if text_column_indices is None:
        text_column_indices = [0]
    all_text = []
    try:
        df = pd.read_csv(
            filepath, sep='\t', header=None, on_bad_lines='warn',
            quoting=3, low_memory=False, dtype=str
        )
        for col_index in text_column_indices:
            if col_index < df.shape[1]:
                all_text.extend(df.iloc[:, col_index].fillna('').tolist())
            else:
                print(
                    f"Warning: Column index {col_index} is out of bounds for file {filepath}"
                )
        return '\n'.join(filter(None, all_text)) if all_text else None
    except pd.errors.ParserError as pe:
        print(f"Parser error while reading TSV {filepath}: {pe}. Skipping.")
        return None
    except Exception as e:
        print(f"Error reading .tsv file {filepath}: {e}")
        return None


def read_pdf_file(filepath: str) -> str | None:
    """
    Reads text from a .pdf file using PyMuPDF.

    Args:
        filepath: Path to the file.

    Returns:
        The extracted text, or None if an error occurs.
    """
    try:
        doc = fitz.open(filepath)
        full_text = "".join(page.get_text("text") for page in doc)
        doc.close()
        full_text = re.sub(r'-\n', '', full_text)
        full_text = re.sub(r'\s*\n\s*', ' ', full_text).strip()
        return full_text
    except Exception as e:
        print(f"Error reading .pdf file {filepath}: {e}")
        return None


def get_text_from_file(filepath: str) -> str | None:
    """
    Determines the file type by extension and calls the appropriate reader.

    Args:
        filepath: Path to the file.

    Returns:
        The extracted text, or None if the format is unsupported or an error occurs.
    """
    _, extension = os.path.splitext(filepath)
    extension = extension.lower()
    readers = {
        '.txt': read_text_file,
        '.docx': read_docx_file,
        '.tsv': read_tsv_file,
        '.pdf': read_pdf_file
    }
    reader_func = readers.get(extension)

    if reader_func:
        if extension == '.tsv':
            return reader_func(filepath, text_column_indices=[0])
        return reader_func(filepath)
    else:
        print(f"Unsupported file format: {extension} for {filepath}")
        return None


# --- 4. Text Analysis Function ---
def analyze_motivation(text: str) -> list[dict]:
    """
    Analyzes the input text to identify “approach–avoidance” motivational patterns.

    The function splits the text into sentences, preprocesses them (lemmatization,
    removal of stop-words and punctuation), searches for keywords, detects negations,
    and classifies sentences.

    Args:
        text: The input text to analyze.

    Returns:
        A list of dictionaries, one for each non-neutral sentence, containing:
        "Sentence", "Classification", "Keywords Found", "Negated Keywords Found".
    """
    if not text or not text.strip():
        print("Warning: Input text is empty or consists only of whitespace.")
        return []

    try:
        initial_sentences = sent_tokenize(text)
    except Exception as e:
        print(f"Sentence tokenization error: {e}. Processing as a single sentence.")
        initial_sentences = [text]

    sentences = [s.strip() for s in initial_sentences if s.strip()]
    num_initial_sentences = len(initial_sentences)
    num_non_empty_sentences = len(sentences)

    results = []
    print("\nINFO FOR PROCESSING (VOLUME METRICS – ANALYSIS STAGE):")
    print(f"Extracted and initially segmented: {num_initial_sentences} sentences.")
    print(f"Non-empty sentences to be analyzed: {num_non_empty_sentences}.")

    for sentence_stripped in sentences:
        tokens_original_lower = word_tokenize(sentence_stripped.lower())
        if not tokens_original_lower:
            continue

        lemmatized_tokens_filtered = []
        original_token_indices_for_lemmas = []

        for original_idx, token in enumerate(tokens_original_lower):
            if (token not in stop_words_english and
                    token not in punctuation_set and
                    len(token) > 1 and
                    not token.isdigit() and
                    any(c.isalpha() for c in token)):
                lemma = lemmatizer.lemmatize(token)
                lemma_verb = lemmatizer.lemmatize(token, pos='v')
                final_lemma = lemma_verb if len(lemma_verb) < len(lemma) else lemma
                lemmatized_tokens_filtered.append(final_lemma)
                original_token_indices_for_lemmas.append(original_idx)

        if not lemmatized_tokens_filtered:
            continue

        found_approach_lemmas = set()
        found_avoidance_lemmas = set()
        negated_keyword_lemmas = set()

        for lemma_idx, lemma in enumerate(lemmatized_tokens_filtered):
            is_approach = lemma in APPROACH_KEYWORDS
            is_avoidance = lemma in AVOIDANCE_KEYWORDS

            if is_approach or is_avoidance:
                is_negated_flag = False
                original_token_idx = original_token_indices_for_lemmas[lemma_idx]
                window_start_idx = max(0, original_token_idx - NEGATION_WINDOW)

                for k_idx in range(window_start_idx, original_token_idx):
                    if tokens_original_lower[k_idx] in NEGATION_WORDS:
                        is_negated_flag = True
                        negated_keyword_lemmas.add(lemma)
                        break
                if not is_negated_flag:
                    if is_approach:
                        found_approach_lemmas.add(lemma)
                    if is_avoidance:
                        found_avoidance_lemmas.add(lemma)

        classification = "Neutral"
        final_keywords_list = []

        if found_approach_lemmas and found_avoidance_lemmas:
            classification = "Mixed"
            final_keywords_list = sorted(list(found_approach_lemmas | found_avoidance_lemmas))
        elif found_approach_lemmas:
            classification = "Approach"
            final_keywords_list = sorted(list(found_approach_lemmas))
        elif found_avoidance_lemmas:
            classification = "Avoidance"
            final_keywords_list = sorted(list(found_avoidance_lemmas))

        if classification != "Neutral":
            results.append({
                "Sentence": sentence_stripped,
                "Classification": classification,
                "Keywords Found": ", ".join(final_keywords_list),
                "Negated Keywords Found": ", ".join(sorted(list(negated_keyword_lemmas)))
            })
    return results


# --- 5. Report-Generation Function ---
def generate_report(analysis_results: list[dict], output_filepath: str):
    """
    Builds a CSV report from the analysis results.

    Args:
        analysis_results: A list of dictionaries returned by analyze_motivation.
        output_filepath:  Destination path for the CSV file.
    """
    if not analysis_results:
        print("No data to generate a report (no non-neutral sentences found).")
        return

    df_report = pd.DataFrame(analysis_results)
    if "Negated Keywords Found" not in df_report.columns:
        df_report["Negated Keywords Found"] = ""

    df_report = df_report[
        ["Sentence", "Classification", "Keywords Found", "Negated Keywords Found"]
    ]
    try:
        df_report.to_csv(output_filepath, index=False,
                         encoding='utf-8-sig', sep=';')
        print(f"Report successfully written: {output_filepath}")
    except Exception as e:
        print(f"Error writing report to {output_filepath}: {e}")


# --- 6. Visualisation Function ---
def visualize_analysis_results(report_filepath: str,
                               num_segments: int = 5,
                               top_n_keywords: int = 15):
    """
    Builds and displays charts from the CSV report.

    Args:
        report_filepath: Path to the CSV report.
        num_segments:    How many equal segments to split the interview into.
        top_n_keywords:  How many keywords to show in the “top-N” charts.
    """
    print(f"\n--- Creating visualisations from: {report_filepath} ---")
    try:
        df_viz = pd.read_csv(report_filepath, sep=';')
    except FileNotFoundError:
        print(f"Error: report file not found at {report_filepath}")
        return
    except Exception as e:
        print(f"Error reading report {report_filepath}: {e}")
        return

    if df_viz.empty:
        print("Report is empty; no visualisations will be created.")
        return

    # -------- Figure 2. Class distribution --------
    print("\nINFO (FIGURE 2 – CLASS DISTRIBUTION):")
    classification_counts = df_viz['Classification'].value_counts()
    print("Sentence count by class:")
    for cls, count_val in classification_counts.items():
        print(f"- '{cls}': {count_val} sentence(s)")
    total_classified_sentences_viz = classification_counts.sum()
    print(f"Total non-neutral sentences in report: {total_classified_sentences_viz}")
    if total_classified_sentences_viz > 0:
        for cls, count_val in classification_counts.items():
            percentage = (count_val / total_classified_sentences_viz) * 100
            print(f"- Share of '{cls}': {percentage:.2f}%")

    plt.figure(figsize=(8, 6))
    sns.countplot(
        x='Classification', data=df_viz, palette='viridis',
        order=df_viz['Classification'].value_counts().index
    )
    plt.title('Figure 2. Distribution of sentence classifications')
    plt.xlabel('Classification type')
    plt.ylabel('Sentence count')
    plt.tight_layout()
    plt.show()

    # -------- Helper for keyword tallies --------
    def get_keyword_counts_from_series(keyword_series: pd.Series) -> Counter:
        all_kws = []
        for k_str in keyword_series.dropna().astype(str):
            if k_str:
                keywords_in_sentence = [
                    k.strip() for k in k_str.split(',') if k.strip()
                ]
                all_kws.extend(keywords_in_sentence)
        return Counter(all_kws)

    # -------- Figure 3. Top non-negated keywords --------
    found_kw_counts = get_keyword_counts_from_series(df_viz['Keywords Found'])
    if found_kw_counts:
        top_found_kws = found_kw_counts.most_common(top_n_keywords)
        if top_found_kws:
            print(f"\nINFO (FIGURE 3 – TOP {top_n_keywords} NON-NEGATED KEYWORDS):")
            for kw, count_val in top_found_kws:
                print(f"- '{kw}': {count_val} time(s)")
            plt.figure(figsize=(12, 8))
            top_found_df_viz = pd.DataFrame(
                top_found_kws, columns=['Keyword', 'Count'])
            sns.barplot(x='Count', y='Keyword',
                        data=top_found_df_viz, palette='magma')
            plt.title(f'Figure 3. Top-{top_n_keywords} non-negated keywords')
            plt.xlabel('Frequency')
            plt.ylabel('Keyword')
            plt.tight_layout()
            plt.show()
        else:
            print("No non-negated keywords found for Figure 3.")
    else:
        print("Unable to extract non-negated keywords for Figure 3.")

    # -------- Figure 4. Top negated keywords --------
    if 'Negated Keywords Found' in df_viz.columns:
        negated_kw_counts = get_keyword_counts_from_series(
            df_viz['Negated Keywords Found'])
        if negated_kw_counts:
            top_negated_kws = negated_kw_counts.most_common(top_n_keywords)
            if top_negated_kws:
                print(f"\nINFO (FIGURE 4 – TOP {top_n_keywords} NEGATED KEYWORDS):")
                for kw, count_val in top_negated_kws:
                    print(f"- '{kw}': {count_val} time(s)")
                plt.figure(figsize=(12, 8))
                top_negated_df_viz = pd.DataFrame(
                    top_negated_kws, columns=['Keyword', 'Count'])
                sns.barplot(x='Count', y='Keyword',
                            data=top_negated_df_viz, palette='coolwarm')
                plt.title(f'Figure 4. Top-{top_n_keywords} negated keywords')
                plt.xlabel('Frequency')
                plt.ylabel('Keyword')
                plt.tight_layout()
                plt.show()
            else:
                print("No negated keywords found for Figure 4.")
        else:
            print("Unable to extract negated keywords for Figure 4 (empty column?).")
    else:
        print("Column 'Negated Keywords Found' missing; Figure 4 skipped.")

    # -------- Figure 5. Dynamics by segments --------
    if len(df_viz) >= num_segments * 2 and num_segments > 0:
        segment_len = len(df_viz) // num_segments
        if segment_len == 0:
            segment_len = 1
            actual_num_segments = len(df_viz)
        else:
            actual_num_segments = num_segments

        df_viz_copy = df_viz.copy()
        df_viz_copy['Segment'] = 0
        current_segment_num = 1
        for i in range(0, len(df_viz_copy), segment_len):
            if current_segment_num > actual_num_segments:
                df_viz_copy.loc[i:, 'Segment'] = actual_num_segments
                break
            end_idx = min(i + segment_len, len(df_viz_copy))
            df_viz_copy.loc[i:end_idx - 1, 'Segment'] = current_segment_num
            current_segment_num += 1
        if df_viz_copy['Segment'].iloc[-1] == 0:
            df_viz_copy.loc[df_viz_copy['Segment'] == 0,
                            'Segment'] = actual_num_segments

        segment_class_counts = df_viz_copy.groupby(
            ['Segment', 'Classification']).size().unstack(fill_value=0)

        if not segment_class_counts.empty and not segment_class_counts.sum(axis=1).eq(0).all():
            segment_proportions_viz = segment_class_counts.apply(
                lambda x: x * 100 / sum(x) if sum(x) > 0 else 0, axis=1
            )
            print(
                f"\nINFO (FIGURE 5 – DYNAMICS ACROSS {actual_num_segments} SEGMENTS):")
            print("Percentage breakdown per segment:")
            print(segment_proportions_viz.to_string())

            segment_proportions_viz.plot(
                kind='bar', stacked=True, figsize=(12, 7), colormap='tab10'
            )
            plt.title(
                f'Figure 5. Dynamics of classification shares over {actual_num_segments} segments')
            plt.xlabel('Interview segment')
            plt.ylabel('Percentage (%)')
            segment_labels = sorted(df_viz_copy['Segment'].unique())
            plt.xticks(ticks=range(len(segment_labels)),
                       labels=segment_labels, rotation=0)
            plt.legend(title='Classification',
                       bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.show()
        else:
            print("Not enough data to build Figure 5 (segment dynamics).")
    else:
        print(f"Not enough sentences ({len(df_viz)}) to split into "
              f"{num_segments} segments (Figure 5).")

# --- 7. Visualisation Function for comparisons of two interviews ---
def visualize_analysis_results_dual(report_filepath1: str,
                                   report_filepath2: str,
                                   num_segments: int = 5,
                                   top_n_keywords: int = 15):
    """Draws every figure twice (left/right) so you can compare two reports."""

    # —— 1. File reader ——
    def _load_report(path: str):
        try:
            df = pd.read_csv(path, sep=';')
        except Exception as e:
            print(f"[{os.path.basename(path)}] Error reading report: {e}")
            return None
        if df.empty:
            print(f"[{os.path.basename(path)}] Report is empty – skipping.")
            return None
        return df

    left_df  = _load_report(report_filepath1)
    right_df = _load_report(report_filepath2)
    if left_df is None or right_df is None:
        return

    # —— 2. Class distribution ——
    print("\n=== FIGURE 2 – CLASS DISTRIBUTION (side‑by‑side) ===")
    for label, df in zip(
            [os.path.basename(report_filepath1), os.path.basename(report_filepath2)],
            [left_df, right_df]):
        class_counts = df['Classification'].value_counts()
        total = class_counts.sum()
        print(f"\n{label}:")
        for cls, cnt in class_counts.items():
            pct = cnt * 100 / total if total else 0
            print(f"  – {cls}: {cnt} sent. ({pct:.2f} %)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.countplot(ax=axes[0], x='Classification', data=left_df, palette='viridis',
                  order=left_df['Classification'].value_counts().index)
    axes[0].set_title(os.path.basename(report_filepath1))
    axes[0].set_xlabel('Classification')
    axes[0].set_ylabel('Sentence count')

    sns.countplot(ax=axes[1], x='Classification', data=right_df, palette='viridis',
                  order=right_df['Classification'].value_counts().index)
    axes[1].set_title(os.path.basename(report_filepath2))
    axes[1].set_xlabel('Classification')
    axes[1].set_ylabel('')

    fig.suptitle('Figure 2. Distribution of sentence classifications')
    fig.tight_layout()
    plt.show()

    # —— Helper for keyword tallies ——
    def _kw_counts(df: pd.DataFrame, column: str):
        kws = []
        for cell in df[column].dropna().astype(str):
            kws.extend([w.strip() for w in cell.split(',') if w.strip()])
        return Counter(kws)

    # —— 3. Top non‑negated keywords ——
    left_kw  = _kw_counts(left_df, 'Keywords Found')
    right_kw = _kw_counts(right_df, 'Keywords Found')
    if left_kw or right_kw:
        print("\n=== FIGURE 3 – TOP NON‑NEGATED KEYWORDS (side‑by‑side) ===")
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        for ax, kw_counter, label in zip(
                axes,
                [left_kw, right_kw],
                [os.path.basename(report_filepath1), os.path.basename(report_filepath2)]):
            top = kw_counter.most_common(top_n_keywords)
            print(f"\n{label}:")
            for kw, cnt in top:
                print(f"  – {kw}: {cnt}")
            if top:
                df_kw = pd.DataFrame(top, columns=['Keyword', 'Count'])
                sns.barplot(ax=ax, x='Count', y='Keyword', data=df_kw, palette='magma')
            ax.set_title(label)
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Keyword')
        fig.suptitle(f'Figure 3. Top-{top_n_keywords} non‑negated keywords')
        fig.tight_layout()
        plt.show()

    # —— 4. Top negated keywords ——
    has_neg_left  = 'Negated Keywords Found' in left_df.columns
    has_neg_right = 'Negated Keywords Found' in right_df.columns
    if has_neg_left or has_neg_right:
        print("\n=== FIGURE 4 – TOP NEGATED KEYWORDS (side‑by‑side) ===")
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        used = False
        for ax, df, label, has_neg in zip(
                axes,
                [left_df, right_df],
                [os.path.basename(report_filepath1), os.path.basename(report_filepath2)],
                [has_neg_left, has_neg_right]):
            if not has_neg:
                ax.axis('off')
                continue
            kw_counter = _kw_counts(df, 'Negated Keywords Found')
            top = kw_counter.most_common(top_n_keywords)
            print(f"\n{label}:")
            for kw, cnt in top:
                print(f"  – {kw}: {cnt}")
            if top:
                df_kw = pd.DataFrame(top, columns=['Keyword', 'Count'])
                sns.barplot(ax=ax, x='Count', y='Keyword', data=df_kw, palette='coolwarm')
                ax.set_title(label)
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Keyword')
                used = True
        if used:
            fig.suptitle(f'Figure 4. Top-{top_n_keywords} negated keywords')
            fig.tight_layout()
            plt.show()

    # —— 5. Dynamics by segments ——
    min_required = num_segments * 2
    if len(left_df) >= min_required and len(right_df) >= min_required:
        def _segment_props(df):
            seg_len = len(df) // num_segments or 1
            df_c = df.copy()
            df_c['Segment'] = df_c.index // seg_len + 1
            df_c.loc[df_c['Segment'] > num_segments, 'Segment'] = num_segments
            counts = df_c.groupby(['Segment', 'Classification']).size().unstack(fill_value=0)
            return counts.apply(lambda x: x * 100 / sum(x) if sum(x) else 0, axis=1)

        left_props  = _segment_props(left_df)
        right_props = _segment_props(right_df)
        print(f"\n=== FIGURE 5 – DYNAMICS ACROSS {num_segments} SEGMENTS ===")
        print("\nLeft file proportions:\n", left_props.to_string())
        print("\nRight file proportions:\n", right_props.to_string())

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        left_props.plot(kind='bar', stacked=True, ax=axes[0], colormap='tab10', legend=False)
        right_props.plot(kind='bar', stacked=True, ax=axes[1], colormap='tab10', legend=False)
        axes[0].set_title(os.path.basename(report_filepath1))
        axes[0].set_xlabel('Segment')
        axes[0].set_ylabel('%')
        axes[1].set_title(os.path.basename(report_filepath2))
        axes[1].set_xlabel('Segment')
        axes[1].set_ylabel('%')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Classification', loc='upper center', ncol=len(labels))
        fig.suptitle(f'Figure 5. Dynamics of classification shares over {num_segments} segments')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()
    else:
        print(f"Not enough data in both reports to build Figure 5 (need at least {min_required} sentences each).")

# --- 8. Main Script Logic  ---
if __name__ == "__main__":
    import sys
    INPUT_DIRECTORY = "Text files"

    # 1) Ask for the *first* file
    filename_to_analyse = input("Enter the name or path of the FIRST file to analyse: ").strip()
    if not filename_to_analyse:
        print("No file name provided. Exiting.")
        sys.exit(0)

    # 1a) Optional second file
    second_filename_to_analyse = input("Enter the SECOND file (leave blank to skip): ").strip()

    # 2) Ask whether visualisations are wanted
    visualise_choice = input("Generate visualisations? (y/n): ").strip().lower()
    do_visualise = visualise_choice in {"y", "yes"}

    def _full(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(INPUT_DIRECTORY, path)
    def _process(path: str):
        full_path = _full(path)
        if not os.path.exists(full_path):
            print(f"Error: input file not found at {full_path}")
            return None

        print(f"\n--- Starting analysis for file: {path} ---")
        print(f"Reading file: {full_path} ...")
        text_content = get_text_from_file(full_path)
        if not text_content:
            print(f"Failed to extract text from file: {full_path}")
            return None

        print("Text successfully extracted. Running NLP analysis ...")
        analysed_data = analyze_motivation(text_content)
        num_analyzed_sentences = sum(1 for s in sent_tokenize(text_content) if s.strip())
        num_classified_sentences = len(analysed_data)

        print("\nINFO (OVERALL VOLUME METRICS):")
        print(f"Non-empty sentences analysed: {num_analyzed_sentences}")
        print(f"Classified as Approach / Avoidance / Mixed: {num_classified_sentences} sentence(s).")

        if not analysed_data:
            print("No relevant segments found; nothing to report.")
            return None

        base_name = os.path.splitext(os.path.basename(path))[0]
        clean_base = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)
        report_name = f"analysis_report_{clean_base}.csv"
        generate_report(analysed_data, report_name)
        return report_name

    # Process first file
    report1 = _process(filename_to_analyse)
    if report1 is None:
        sys.exit(1)

    if second_filename_to_analyse:  # Dual‑file path
        report2 = _process(second_filename_to_analyse)
        if report2 is None:
            sys.exit(1)

        # Visualise
        if do_visualise:
            visualize_analysis_results_dual(report1, report2, num_segments=5, top_n_keywords=15)
    else:  # Fallback to original behaviour
        if do_visualise:
            visualize_analysis_results(report1, num_segments=5, top_n_keywords=15)