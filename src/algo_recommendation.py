easy_list= ["images resized in browser number", "empty src tag number", "images downloazded not displayed number",
            "total min gains", "total size to optimize", "viewport", "fontDisplay","usesResponsiveImages",
            "unusedCssRules","unusedJavscript","usesOptimizedImages","modernImageFormats","efficientAnimatedContent",
            "noDocumentWrite"]

medium_list= ["cache header ratio","compress ratio","inline style sheets number","inline js scripts number",
              "error number","js validate", "percent minified css","percent minified js","redirect number",
              "plugins number","print style sheets number","number social network button","style sheets number",
              "eTags Ratio","total Fonts Size","bootupTime","thirdPartySummary","lcpLazyLoaded","domSize","offscreenImages",
              "usesTextCompression","usesHttp2","legacyJavascript","usesPassiveEventListeners"]


hard_list= ["domains number","number of requests","max cookies length","total cookies size","serverResponsetime",
"mainthreadWorkBreakdown","usesLongCacheTtl","totalByteWeight"]

#this is done for a random row, next step: give the id and
#urlscore3= urlscore3[urlscore3["id"]]== id)


features= urlscore3.iloc[5:6,0:45]
list_of_class= urlscore3.iloc[5:6,45:54]

Classe1 = []
Classe2 = []
Classe3 = []

First_Recommendation = []
Second_Recommendation = []


for metric in list_of_class:
    value = list_of_class.at[5, metric]
    
    if value == 3.0:
        Classe3.append(metric)
    if value == 2.0:
        Classe2.append(metric)
    if value == 1.0:
        Classe1.append(metric)

print(Classe1)
print(Classe2)
print(Classe3)


def get_list_importance(easy,medium,hard,feature_importance_list ):
    
    for feat in feature_importance_list:
    
        #retrieve its corelation list
        corr_target = urlscore2.corr()[feat]
        corr = correlation[feat]
        choose = corr[corr > 0.6] #correlation percentage set to 0.6

        #add the easy features
        if feat in easy_list:
            easy.append(feat)

        #add the corraleted features 
        for indice in choose:
            if indice in easy_list:
                easy.append(indice)


        #add the medium features
        if feat in medium_list:
            medium.append(feat)

        #add the medium corraleted features 
        for indice in choose:
            if indice in medium_list:
                medium.append(indice)


        #add the hard features
        if feat in hard_list:
            hard.append(feat)

        #add the hard corraleted features 
        for indice in choose:
            if indice in hard_list:
                hard.append(indice)
                
    return easy, medium, hard

#DomSIze

easy_dom = []
medium_dom = []
hard_dom = []


easy_dom, medium_dom, hard_dom = get_list_importance(easy_dom,medium_dom,hard_dom,importance_domSize_list)


#nbRequest

easy_nb = []
medium_nb = []
hard_nb = []

easy_nb, easy_nb, easy_nb = get_list_importance(easy_nb,medium_nb,hard_nb,importance_nbRequest_list)


#ResponsiveSize
easy_responsesSize = []
medium_responsesSize = []
hard_responsesSize = []

easy_responsesSize, medium_responsesSize, easy_responsesSize = get_list_importance(easy_responsesSize,medium_responsesSize,hard_responsesSize,importances_responsesSize_list)


#totalBlockingTime
easy_totalBlockingTime = []
medium_totalBlockingTime = []
hard_totalBlockingTime = []

easy_totalBlockingTime, easy_totalBlockingTime, easy_totalBlockingTime = get_list_importance(easy_totalBlockingTime,medium_totalBlockingTime,hard_totalBlockingTime,importances_totalBlockingTime_list)

#largestContentfulPaint
#largestContentfulPaint
easy_largestContentfulPaint = []
medium_largestContentfulPaint = []
hard_largestContentfulPaint =[]

easy_largestContentfulPaint, medium_largestContentfulPaint, hard_largestContentfulPaint = get_list_importance(easy_largestContentfulPaint, medium_largestContentfulPaint, hard_largestContentfulPaint,importance_largestContentfulPaint_list) 

#cumulativeLayoutShift 
easy_cumulativeLayoutShift =[]
medium_cumulativeLayoutShift = []
hard_cumulativeLayoutShift =[]

easy_cumulativeLayoutShift, medium_cumulativeLayoutShift, hard_cumulativeLayoutShift = get_list_importance(easy_cumulativeLayoutShift, medium_cumulativeLayoutShift, hard_cumulativeLayoutShift,importance_cumulativeLayoutShift_list)

#firstContentfulPaint
easy_firstContentfulPaint =[]
medium_firstContentfulPaint = []
hard_firstContentfulPaint =[]

easy_firstContentfulPaint, medium_firstContentfulPaint, hard_firstContentfulPaint = get_list_importance(easy_firstContentfulPaint, medium_firstContentfulPaint, hard_firstContentfulPaint,importances_firstContentfulPaint_list)

#speedIndex
easy_speedIndex =[]
medium_speedIndex = []
hard_speedIndex =[]

easy_speedIndex, medium_speedIndex, hard_speedIndex = get_list_importance(easy_speedIndex, medium_speedIndex, hard_speedIndex,importances_speedIndext_list)

#interactive
easy_interactive =[]
medium_interactive = []
hard_interactive =[]


easy_interactive, medium_interactive, hard_interactive = get_list_importance(easy_interactive, medium_interactive, hard_interactive,importances_interactive_list)

#in the order of ponderation's weight
        
def append_recommendation(recommendation_list,classe):
    

    for metric in classe:
        

        if metric == "classe_domSize":

            #easy
            for easy in easy_dom:
                recommendation_list.append(easy)
            #medium
            for medium in medium_dom:
                recommendation_list.append(medium)
            #hard
            for hard in hard_dom:
                recommendation_list.append(hard)


        if metric == "classe_nbRequest":

            #easy
            for easy in easy_nb:
                recommendation_list.append(easy)
                
            #medium
            for medium in medium_nb:
                recommendation_list.append(medium)

            #hard
            for hard in hard_nb:
                recommendation_list.append(hard)
        
        if metric == "classe_responsesSize":
            #easy
            for easy in easy_responsesSize:
                recommendation_list.append(easy)
                
            #medium
            for medium in medium_responsesSize:
                recommendation_list.append(medium)

            #hard
            for hard in hard_responsesSize:
                recommendation_list.append(hard)
                
        if metric == "classe_totalBlockingTime":
            for easy in easy_totalBlockingTime:
                recommendation_list.append(easy)
                
            #medium
            for medium in medium_totalBlockingTime:
                recommendation_list.append(medium)

            #hard
            for hard in hard_totalBlockingTime:
                recommendation_list.append(hard)
                
        if metric == "classe_largestContentfulPaint":
            #easy
            for easy in easy_largestContentfulPaint:
                recommendation_list.append(easy)
                
            #medium
            for medium in medium_largestContentfulPaint:
                recommendation_list.append(medium)

            #hard
            for hard in hard_largestContentfulPaint:
                recommendation_list.append(hard)
        
        if metric == "classe_cumulativeLayoutShift":
            #easy
            for easy in easy_cumulativeLayoutShift:
                recommendation_list.append(easy)
                
            #medium
            for medium in medium_cumulativeLayoutShift:
                recommendation_list.append(medium)

            #hard
            for hard in hard_cumulativeLayoutShift:
                recommendation_list.append(hard)
        
        if metric == "classe_firstContentfulPaint":
            #easy
            for easy in easy_firstContentfulPaint:
                recommendation_list.append(easy)
                
            #medium
            for medium in medium_firstContentfulPaint:
                recommendation_list.append(medium)

            #hard
            for hard in hard_firstContentfulPaint:
                recommendation_list.append(hard)
        
        if metric == "classe_speedIndex":
            #easy
            for easy in easy_speedIndex:
                recommendation_list.append(easy)
                
            #medium
            for medium in medium_speedIndex:
                recommendation_list.append(medium)

            #hard
            for hard in hard_speedIndex:
                recommendation_list.append(hard)
        
       
        if metric == "classe_interactive":
            #easy
            for easy in easy_interactive:
                recommendation_list.append(easy)
                
            #medium
            for medium in medium_interactive:
                recommendation_list.append(medium)

            #hard
            for hard in hard_interactive:
                recommendation_list.append(hard)
       
    
    
    
 def display_recommendation(recommendation_list):
    
    print("Recommendations:")
    
    for i in range(len(recommendation_list)):
        
        print(i, recommendation_list[i])


append_recommendation(First_Recommendation,Classe1)
First_Recommendation = list(OrderedDict.fromkeys(First_Recommendation))
display_recommendation(First_Recommendation)
