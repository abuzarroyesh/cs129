Final\_project
================

### Covid data

``` r
library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.0 ──

    ## ✓ ggplot2 3.3.2     ✓ purrr   0.3.4
    ## ✓ tibble  3.0.4     ✓ dplyr   1.0.2
    ## ✓ tidyr   1.1.2     ✓ stringr 1.4.0
    ## ✓ readr   1.4.0     ✓ forcats 0.5.0

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
library(readxl)

covid_raw <- 
  read_csv("data/10-20-2020.csv") %>% 
  filter(Country_Region == "US") %>% 
  select(
    fips = FIPS, 
    county_name = Admin2, 
    state = Province_State, 
    confirmed = Confirmed, 
    deaths = Deaths
  ) %>% 
  mutate(fips = as.integer(fips)) %>% 
  filter(!str_detect(county_name, "Unassigned|Out of")) %>% 
  filter(!is.na(fips)) %>% 
  filter(!state == "Puerto Rico")
```

    ## 
    ## ── Column specification ────────────────────────────────────────────────────────
    ## cols(
    ##   FIPS = col_double(),
    ##   Admin2 = col_character(),
    ##   Province_State = col_character(),
    ##   Country_Region = col_character(),
    ##   Last_Update = col_datetime(format = ""),
    ##   Lat = col_double(),
    ##   Long_ = col_double(),
    ##   Confirmed = col_double(),
    ##   Deaths = col_double(),
    ##   Recovered = col_double(),
    ##   Active = col_double(),
    ##   Combined_Key = col_character(),
    ##   Incidence_Rate = col_double(),
    ##   `Case-Fatality_Ratio` = col_double()
    ## )

### Demographic data

``` r
data <- 
  read_tsv("data/pcen_v2019_y19.txt", col_names = FALSE) %>% 
  mutate(
    series = str_sub(X1, start = 1L, end = 4L), 
    estimate_year = str_sub(X1, start = 5L, end = 8L), 
    estimate_month = str_sub(X1, start = 9L, end = 9L), 
    fips = str_sub(X1, start = 10L, end = 14L),
    # county_fips = str_sub(X1, start = 12L, end = 14L), 
    age = str_sub(X1, start = 15L, end = 16L),
    race_sex = str_sub(X1, start = 17L, end = 17L), 
    hispanic_origin = str_sub(X1, start = 18L, end = 18L),
    population = str_sub(X1, start = 19L, end = 26L),
    race_sex = 
      recode(
        race_sex, 
        "1" = "m white", 
        "2" = "f white", 
        "3" = "m black", 
        "4" = "f black", 
        "5" = "m native", 
        "6" = "f native", 
        "7" = "m asian", 
        "8" = "f asian"
      ),
    hispanic_origin = if_else(hispanic_origin == 1, "non-hispanic", "hispanic"),
    sex = str_sub(race_sex, start = 1L, end = 1L),
    race = str_sub(race_sex, start = 3L)
  ) %>% 
  mutate_at(vars(fips, age, population), as.integer) %>% 
  mutate(
    age_bracket = 
      case_when(
        age >= 0 & age <= 4   ~ "age1",
        age >= 5 & age <= 17  ~ "age2",
        age >= 18 & age <= 29 ~ "age3",
        age >= 30 & age <= 39 ~ "age4",
        age >= 40 & age <= 49 ~ "age5",
        age >= 50 & age <= 64 ~ "age6",
        age >= 65 & age <= 74 ~ "age7",
        age >= 75 & age <= 84 ~ "age8",
        age >= 85             ~ "age9"
      )
  ) %>% 
  select(-X1, -race_sex, -age) 
```

    ## 
    ## ── Column specification ────────────────────────────────────────────────────────
    ## cols(
    ##   X1 = col_character()
    ## )

What age means: “age1” = “age0-4” “age2” = “age5-17” “age3” = “age18-29”
“age4” = “age30-39” “age5” = “age40-49” “age6” = “age50-64” “age7” =
“age65-74” “age8” = “age75-84” “age9” = “age85+”

``` r
##Writing function to extract percentage for each category
summary_stats <- function(variable_name){
  variable_name = enquo(variable_name)
  
  data %>% 
  group_by(fips, !!variable_name) %>% 
  summarize(population = sum(population), .groups = "drop") %>% 
  group_by(fips) %>% 
  mutate(prop = population / sum(population)) %>% 
  select(-population) %>% 
  spread(key = !!variable_name, value = prop)
}

##Running the previous function on all categories
demographics <- 
  summary_stats(race) %>% 
  left_join(summary_stats(sex), by = "fips") %>%
  left_join(summary_stats(hispanic_origin), by = "fips") %>% 
  left_join(summary_stats(age_bracket), by = "fips") %>% 
  select(-white, -m, -`non-hispanic`, -`age1`) %>% #Dropping on column from each category
  rename(female = f)
```

### Poverty data

``` r
poverty <- 
  read_xls("data/PovertyEstimates.xls") %>% 
  mutate(fips = as.integer(FIPStxt)) %>% 
  filter(!is.na(Urban_Influence_Code_2013)) %>% 
  select(
    fips, 
    povall = PCTPOVALL_2018, 
    urban_influence = Urban_Influence_Code_2013, 
    pov017 = PCTPOV017_2018, 
    pov517 = PCTPOV517_2018, 
    med_hh_income = MEDHHINC_2018
  )
```

### Education data

``` r
education <- 
  read_xls("data/Education.xls") %>% 
  mutate(fips = as.integer(`FIPS Code`)) %>% 
  filter(!is.na(`2013 Urban Influence Code`)) %>% 
  select(
    fips, 
    #no_diploma = `Percent of adults with less than a high school diploma, 2014-18`,
    #dropping this category since all add up to 1. 
    high_school = `Percent of adults with a high school diploma only, 2014-18`, 
    some_college = `Percent of adults completing some college or associate's degree, 2014-18`, 
    bachelor_plus = `Percent of adults with a bachelor's degree or higher, 2014-18`
  ) 
```

### Unemployment data

``` r
unemp <- 
  read_xls("data/Unemployment.xls") %>% 
  mutate(fips = as.integer(`FIPStxt`)) %>% 
  filter(!is.na(`Urban_influence_code_2013`)) %>% 
  select(
    fips, 
    unemp = Unemployment_rate_2019, 
    labor_force = Civilian_labor_force_2019
  )
```

### Life expectancy, Physical activity, and obesity

``` r
life <- 
  read_xlsx("data/IHME_county_data_LifeExp.xlsx", sheet = 1) %>% 
  filter(!is.na(County)) %>% 
  select(
    county_name = County, 
    state = State,
    female_life_exp = `Female life expectancy, 2010 (years)`, 
    male_life_exp = `Male life expectancy, 2010 (years)`
  ) %>% 
  left_join(
    read_xlsx("data/IHME_county_data_LifeExp.xlsx", sheet = 2) %>% 
      filter(!is.na(County)) %>% 
      select(
        county_name = County, 
        state = State,
        female_phy_act = `Female sufficient physical activity  prevalence, 2011* (%)`, 
        male_phy_act = `Male sufficient physical activity  prevalence, 2011* (%)`
      ),
    by = c("county_name", "state")
  ) %>% 
  left_join(
    read_xlsx("data/IHME_county_data_LifeExp.xlsx", sheet = 3) %>% 
      filter(!is.na(County)) %>% 
      select(
        county_name = County, 
        state = State,
        female_obesity = `Female obesity prevalence, 2011* (%)`, 
        male_obesity = `Male obesity  prevalence, 2011* (%)`
      ), 
    by = c("county_name", "state")
  )
```

    ## New names:
    ## * `` -> ...11
    ## New names:
    ## * `` -> ...11

### Diabetes

``` r
diabetes <- 
  read_xlsx("data/IHME_county_data_Diabetes.xlsx", sheet = "Total") %>% 
  select(
    fips = FIPS, 
    female_diabetes = `Prevalence, 2012, Females`, 
    male_diabetes = `Prevalence, 2012, Males`
  )
```

### Mortality rates

``` r
mortality <- 
  read_tsv("data/Multiple Cause of Death, 1999-2018.txt", n_max = 17189) %>% 
  mutate(
    fips = as.integer(`County Code`),
    mort_rate = Deaths / Population,
    age = 
      recode(
        `Ten-Year Age Groups`, 
        "< 1 year" = "mort0", 
        "1-4 years" = "mort1",
        "5-14 years" = "mort2",
        "15-24 years" = "mort3",
        "25-34 years" = "mort4",
        "35-44 years" = "mort5",
        "45-54 years" = "mort6",
        "55-64 years" = "mort7",
        "65-74 years" = "mort8",
        "75-84 years" = "mort9",
        "85+ years" = "mort10"
      )
    ) %>% 
  filter(age != "Not Stated") %>% 
  select(fips, age, mort_rate) %>% 
  spread(key = age, value = mort_rate) %>% 
  mutate_all(replace_na, 0)
```

    ## 
    ## ── Column specification ────────────────────────────────────────────────────────
    ## cols(
    ##   Notes = col_logical(),
    ##   County = col_character(),
    ##   `County Code` = col_character(),
    ##   `Ten-Year Age Groups` = col_character(),
    ##   `Ten-Year Age Groups Code` = col_character(),
    ##   Deaths = col_double(),
    ##   Population = col_double(),
    ##   `Crude Rate` = col_character()
    ## )

    ## Warning: 2 parsing failures.
    ##   row        col expected         actual                                          file
    ## 10207 Population a double Not Applicable 'data/Multiple Cause of Death, 1999-2018.txt'
    ## 12481 Population a double Not Applicable 'data/Multiple Cause of Death, 1999-2018.txt'

### Merging the datasets

``` r
covid_raw %>% 
  left_join(
    read_csv("data/county_populations.csv"), by = c("fips" = "county_code")
  ) %>% 
  left_join(demographics, by = "fips") %>% 
  left_join(poverty, by = "fips") %>% 
  left_join(education, by = "fips") %>% 
  left_join(unemp, by = "fips") %>% 
  left_join(life, c("county_name", "state")) %>% 
  left_join(diabetes, by = "fips") %>% 
  left_join(mortality, by = "fips") %>% 
  write_csv("covid_cleaned.csv")
```

    ## 
    ## ── Column specification ────────────────────────────────────────────────────────
    ## cols(
    ##   county_code = col_double(),
    ##   total_pop = col_double()
    ## )
