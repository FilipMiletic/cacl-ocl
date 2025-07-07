install.packages("reticulate")
library(reticulate)

py_install("pandas")
py_require("pandas")
py_require("pyarrow")
py_config()
pd <- import("pandas")  # Import Python pandas

library(dplyr)
library(caret)
library(effects)
library(relaimpo)
library(jtools)

get_processed_dataframe <- function(feature_df) {
  # Define columns to ignore (text-based columns not needed for analysis)
  columns_to_ignore <- c("abstract", "full_text", "title", "text")
  
  # Drop ignored columns if they exist
  feature_df <- feature_df %>% select(-any_of(columns_to_ignore))
  
  # Drop 'after' and 'acl_id' (not included in scaling)
  feature_df_current <- feature_df %>% select(-any_of(c("after", "acl_id")))
  
  # Identify columns with only one unique value or all NA values
  cols_to_drop <- feature_df_current %>%
    select(where(~ n_distinct(.) == 1 | all(is.na(.)))) %>%
    colnames()
  
  # Drop the identified columns
  feature_df_current <- feature_df_current %>% select(-any_of(cols_to_drop))
  
  # Replace Inf and -Inf with NA, then fill NA values with column means
  feature_df_current <- feature_df_current %>%
    mutate(across(everything(), ~ ifelse(is.infinite(.), NA, .))) %>%
    mutate(across(everything(), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))
  
  # Define columns to be scaled (excluding 'after' and 'acl_id')
  non_scaling_cols <- c("after", "acl_id")
  scaling_features <- setdiff(colnames(feature_df_current), non_scaling_cols)
  
  # Scale numerical features using StandardScaler (centering & scaling)
  preproc <- preProcess(feature_df_current[, scaling_features], method = c("center", "scale"))
  feature_df_scaled <- predict(preproc, feature_df_current[, scaling_features])
  
  # Convert back to a data frame
  feature_df_scaled <- as.data.frame(feature_df_scaled)
  
  # Ensure no NaN values exist in the processed dataframe
  feature_df_scaled[is.na(feature_df_scaled)] <- 0
  
  # Add back 'after' and 'acl_id' if they exist
  for (col in non_scaling_cols) {
    if (col %in% colnames(feature_df)) {
      feature_df_scaled[[col]] <- feature_df[[col]]
    }
  }
  
  return(feature_df_scaled)
}



df_t1 <- pd$read_parquet("/Users/falkne/PycharmProjects/cacl-ocl/cacl_t1_features.parquet")
df_t2 <- pd$read_parquet("/Users/falkne/PycharmProjects/cacl-ocl/cacl_t2_features.parquet")

df_t1$after <- 0
df_t2$after <- 1
t_combined <- rbind(df_t1, df_t2)
t_processed <- get_processed_dataframe(t_combined)

head(t_processed)
most_rel_feats <- c("n_low_intensity_anticipation","n_low_prevalence","n_low_intensity_disgust","avg_aoa","avg_word_length","n_high_aoa","avg_Head_sensorimotor","n_high_Head_sensorimotor","n_VERB_Aspect_Prog","cli","avg_sd_aoa","avg_sd_Head_sensorimotor","n_dependency_auxpass","avg_n_synsets","n_dependency_advcl","avg_n_synsets_noun","n_high_valence","n_polysyllables","n_low_intensity_sadness","avg_arousal","n_dependency_case","n_high_synsets","n_dependency_dobj","smog","n_dependency_nsubjpass","n_high_trust_intensity","n_high_synsets_noun","n_high_anger_intensity","lix","n_high_socialness","n_dependency_poss","n_low_intensity_anger","avg_Mouth_sensorimotor","n_positive_sentiment","avg_Torso_sensorimotor","avg_sd_Mouth_sensorimotor","sentiment_score","avg_dominance","n_high_fear_intensity","n_aux","entropy","n_controversial_socialness","n_PROPN_Number_Plur","n_high_arousal","n_low_synsets_verb","n_monosyllables","avg_intensity_anger","n_long_words","avg_sd_Interoceptive_sensorimotor","avg_intensity_fear","n_low_synsets_noun","avg_sd_concreteness","avg_Foot_leg_sensorimotor","rix","avg_intensity_surprise","avg_Hand_arm_sensorimotor","n_high_Foot_leg_sensorimotor","avg_valence","avg_intensity_trust","n_low_valence","n_dependency_attr","n_low_intensity_surprise","avg_sd_iconicity","avg_concreteness","ari","gunning_fog","n_PRON_Person_2","n_high_surprise_intensity","ID","n_low_intensity_fear","avg_prevalence","mtld","n_det","n_VERB_VerbForm_Inf","n_low_synsets","n_PUNCT_PunctType_Brck","n_DET_Definite_Ind","n_dependency_pcomp","n_dependency_det","n_low_arousal","avg_Interoceptive_sensorimotor","n_high_synsets_adv","lexical_density","n_PUNCT_PunctType_Peri","sichel_s","n_verb","n_high_synsets_verb","n_controversial_Hand_arm_sensorimotor","n_low_Visual_sensorimotor","n_controversial_Foot_leg_sensorimotor","avg_sd_Torso_sensorimotor","n_PUNCT_PunctType_Dash","n_controversial_Haptic_sensorimotor","avg_intensity_sadness","n_PRON_Poss_Yes","lemma_token_ratio","n_controversial_Mouth_sensorimotor","avg_sd_Auditory_sensorimotor","n_VERB_Aspect_Perf","n_NOUN_Number_Plur","avg_intensity_joy","herdan_c","n_VERB_Person_3","n_VERB_Number_Sing","n_low_Haptic_sensorimotor","n_part","n_PUNCT_PunctType_Comm","compressibility","n_ordinal","rttr","cttr","n_dependency_relcl","n_VERB_Mood_Ind","n_high_Torso_sensorimotor","dougast_u","n_global_token_hapax_dislegomena","n_high_Interoceptive_sensorimotor","n_low_Olfactory_sensorimotor","n_global_lemma_hapax_dislegomena","n_low_Gustatory_sensorimotor","avg_sd_Visual_sensorimotor","avg_sd_socialness","n_low_Torso_sensorimotor","n_controversial_Interoceptive_sensorimotor","n_low_Mouth_sensorimotor","n_VERB_VerbForm_Part","n_low_Foot_leg_sensorimotor","n_controversial_iconicity","avg_intensity_anticipation","avg_sd_Foot_leg_sensorimotor","n_low_Auditory_sensorimotor","n_DET_Definite_Def","avg_n_synsets_verb","n_high_prevalence","ttr","n_sentences","n_hapax_legomena","n_dependency_predet","n_PRON_Person_1","n_negative_sentiment","n_low_Hand_arm_sensorimotor","n_low_Interoceptive_sensorimotor","n_controversial_Torso_sensorimotor","avg_Visual_sensorimotor","avg_socialness","n_high_concreteness","msttr","n_characters","n_high_joy_intensity")

model_cacl <- glm(after ~ n_low_intensity_anticipation+n_low_prevalence+n_low_intensity_disgust+avg_aoa+avg_word_length+n_high_aoa+avg_Head_sensorimotor+n_high_Head_sensorimotor+n_VERB_Aspect_Prog+cli+avg_sd_aoa+avg_sd_Head_sensorimotor+n_dependency_auxpass+avg_n_synsets+n_dependency_advcl+avg_n_synsets_noun+n_high_valence+n_polysyllables+n_low_intensity_sadness+avg_arousal+n_dependency_case+n_high_synsets+n_dependency_dobj+smog+n_dependency_nsubjpass+n_high_trust_intensity+n_high_synsets_noun+n_high_anger_intensity+lix+n_high_socialness+n_dependency_poss+n_low_intensity_anger+avg_Mouth_sensorimotor+n_positive_sentiment+avg_Torso_sensorimotor+avg_sd_Mouth_sensorimotor+sentiment_score+avg_dominance+n_high_fear_intensity+n_aux+entropy+n_controversial_socialness+n_PROPN_Number_Plur+n_high_arousal+n_low_synsets_verb+n_monosyllables+avg_intensity_anger+n_long_words+avg_sd_Interoceptive_sensorimotor+avg_intensity_fear+n_low_synsets_noun+avg_sd_concreteness+avg_Foot_leg_sensorimotor+rix+avg_intensity_surprise+avg_Hand_arm_sensorimotor+n_high_Foot_leg_sensorimotor+avg_valence+avg_intensity_trust+n_low_valence+n_dependency_attr+n_low_intensity_surprise+avg_sd_iconicity+avg_concreteness+ari+gunning_fog+n_PRON_Person_2+n_high_surprise_intensity+ID+n_low_intensity_fear+avg_prevalence+mtld+n_det+n_VERB_VerbForm_Inf+n_low_synsets+n_PUNCT_PunctType_Brck+n_DET_Definite_Ind+n_dependency_pcomp+n_dependency_det+n_low_arousal+avg_Interoceptive_sensorimotor+n_high_synsets_adv+lexical_density+n_PUNCT_PunctType_Peri+sichel_s+n_verb+n_high_synsets_verb+n_controversial_Hand_arm_sensorimotor+n_low_Visual_sensorimotor+n_controversial_Foot_leg_sensorimotor+avg_sd_Torso_sensorimotor+n_PUNCT_PunctType_Dash+n_controversial_Haptic_sensorimotor+avg_intensity_sadness+n_PRON_Poss_Yes+lemma_token_ratio+n_controversial_Mouth_sensorimotor+avg_sd_Auditory_sensorimotor+n_VERB_Aspect_Perf+n_NOUN_Number_Plur+avg_intensity_joy+herdan_c+n_VERB_Person_3+n_VERB_Number_Sing+n_low_Haptic_sensorimotor+n_part+n_PUNCT_PunctType_Comm+compressibility+n_ordinal+rttr+cttr+n_dependency_relcl+n_VERB_Mood_Ind+n_high_Torso_sensorimotor+dougast_u+n_global_token_hapax_dislegomena+n_high_Interoceptive_sensorimotor+n_low_Olfactory_sensorimotor+n_global_lemma_hapax_dislegomena+n_low_Gustatory_sensorimotor+avg_sd_Visual_sensorimotor+avg_sd_socialness+n_low_Torso_sensorimotor+n_controversial_Interoceptive_sensorimotor+n_low_Mouth_sensorimotor+n_VERB_VerbForm_Part+n_low_Foot_leg_sensorimotor+n_controversial_iconicity+avg_intensity_anticipation+avg_sd_Foot_leg_sensorimotor+n_low_Auditory_sensorimotor+n_DET_Definite_Def+avg_n_synsets_verb+n_high_prevalence+ttr+n_sentences+n_hapax_legomena+n_dependency_predet+n_PRON_Person_1+n_negative_sentiment+n_low_Hand_arm_sensorimotor+n_low_Interoceptive_sensorimotor+n_controversial_Torso_sensorimotor+avg_Visual_sensorimotor+avg_socialness+n_high_concreteness+msttr+n_characters+n_high_joy_intensity, data=t_processed, family = binomial)
export_summs(model_cacl,
             model.names = c("after GPT"), robust = TRUE, error_pos = c("same"),  error_format = "({conf.low}, {conf.high})",   # Show confidence intervals
             ci_level = 0.95,                              # Use 95% confidence intervals
             bold_signif = 0.05)
result <- summ(model_cacl)
result
write.csv("/Users/falkne/PycharmProjects/cacl-ocl/regression_all.csv")

result2 <- anova(model_cacl)
result2
reduced_model_cacl <- stepAIC(model_cacl)
