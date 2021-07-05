from utils import *
import spacy

nlp1 = spacy.load("trained_model/model-best") #load the best model

samples = [
    "This study used data from the National Education Longitudinal Study (NELS:88) to examine the effects of dual enrollment programs for high school students on college degree attainment.",
    "A number of longitudinal epidemiologic studies, including the Baltimore Longitudinal Study of Aging, the New Mexico Aging Process Study, and the Massachusetts Male Aging Study, have demonstrated age-related increases in the likelihood of developing hypogonadism.",
    "The index comprises two categories, respectively cognitive skill (the latest test results from the Progress in International Reading Literacy Study, PIRLS; the Trends in International Mathematics and Science Study, TIMSS; the Programme for International Student Assessment, PISA; the initial output from the Programme for the International Assessment of Adult Competencies, PIAAC) and educational attainment (the latest literacy rate and graduation rates at the upper secondary and tertiary level).",
    "Comparative Indicators of Education in the United States and Other G-8 Countries: 2009 draws on the most current information about education from four primary sources: the Indicators of National Education Systems (INES) at the Organization for Economic Cooperation and Development (OECD); the Progress in International reading Literacy Study (PIrLS); the Program for International Student Assessment (PISA); and the Trends in International Mathematics and Science Study (TIMSS).",
    "Using secondary data from the 2010 Nielsen Homescan Survey, Rahkovsky and Snyder (2015) reported that regardless of income, consumers shop at 11 different food stores over a year's time. "
]

htmls = []
for s in samples:
    doc = nlp1(s) # input sample text

    html = spacy.displacy.render(doc, style="ent")
    htmls.append(html + "-----<br>")

with open('trained_model.html', 'w') as f:
    f.write("\n".join(htmls))
