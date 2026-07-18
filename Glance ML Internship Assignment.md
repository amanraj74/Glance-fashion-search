# Glance ML Internship Assignment
## Multimodal Fashion & Context Retrieval

---

## 1. Project Overview 

The goal of this assignment is to build an intelligent search engine that can retrieve specific images from a diverse database based on natural language descriptions. The system needs to understand not just "what" someone is wearing, but "where" they are and the "vibe" of their attire. 

## 2. The Dataset 

You are responsible for sourcing or simulating a dataset (minimum 500–1,000 images). The data must contain variations across three primary axes: 

- **Environment:** Office interiors, urban streets, parks, and home settings. 

- **Clothing Types:** Formal (blazers, button-downs), Casual (hoodies, t-shirts), and Outerwear, etc 

- **Color Theory:** A wide palette of garment colors. 

One such dataset is fashionpedia’s <u>dataset or you can download it here</u> 

## 3. Technical Requirements 

Your submission must consist of two distinct workflows, ideally hosted in 1 GitHub repository in different directories or files, clearly defined modules: 

### Part A: The Indexer 

- Feature Extraction: Process the raw images into a searchable format. 

- Vector Storage: Implement a solution to store these representations efficiently (avoiding simple filename keyword matching). 

### Part B: The Retriever (The "Query") 

- Search Logic: Create a script that accepts a natural language string and returns the top $k$ matching images. 

- Context Awareness: The system should handle multi-attribute queries (e.g., color + clothing type + location). 

Focus more on the ML logic aspect of it, rather than the engg work of indexing, for example - if you chose to pick a Vector DB, pick the easiest and most convenient one instead of 

spending time and code on rewriting your own version. You are assessed first and foremost on ML logic. 

**Hint:** While architectures like CLIP provide a strong baseline for zero-shot retrieval, they often struggle with compositionality (e.g., distinguishing "red shirt with blue pants" from "blue shirt with red pants") and fine-grained fashion attributes. The expected solution should be better than vanilla application of CLIP with focus on making it work for fashion based retrieval. 

## 4. Evaluation Queries 

Your system will be judged on its ability to accurately return images for the following prompts: 

1. **Attribute Specific:** _"A person in a bright yellow raincoat."_ 

2. **Contextual/Place:** _"Professional business attire inside a modern office."_ 

3. **Complex Semantic:** _"Someone wearing a blue shirt sitting on a park bench."_ 

4. **Style Inference:** _"Casual weekend outfit for a city walk."_ 

5. **Compositional:** _"A red tie and a white shirt in a formal setting."_ 

## 5. Submission Deliverables 

A single PDF that contains 

1. **Approaches:** Possible ways to solve this problem, tradeoffs, what’s good and when 

2. **Short Write-up on Chosen Approach:** A brief explanation of your chosen architecture & how it handles fashion queries 

3. **Codebase (GitHub) Link:** Clean, documented code for both the indexing and retrieval pipelines. 

4. **Approaches for future work** 

   - a. How to extend this solution for adding locations (cities, places) and weather 

   - b. How to improve precision 

## 6. What We Are Looking For 

#### 1. **Thoughtful solution:** 

   - a. Getting a solution in the age of AI is easy - coding it successfully, understanding the chosen approach (model, arch), what are its shortcomings & how to address that are sought 

   - b. How well does it work for fashion based queries? 

2. **Modular Code:** Is your logic separated from your data? 

3. **Scalability:** Would your retrieval logic work if the dataset grew to 1 million images? 

4. **Zero-Shot Capability:** How well does the system handle descriptions it hasn't seen explicitly in a training label?
