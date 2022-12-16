Today, the internet and the technologies supporting it account for 3.7% of total carbon emissions, which is equivalent to the entire airline industry. Moreover, this number is expected to double in the next 3 years. There is a need for change and impact. As a result, the goal of this research is to find a way to reduce the amount of carbon dioxide emitted from websites. Indeed, the more complex a website is, the more energy it will require to load and the greater its carbon emissions. There is a need to inform developers on what practices to implement in order to reduce their website’s carbon footprint. 

We generated a dataset by analyzing more than 500 websites using two open-source audit tools: GreenIT-analysis and Google Lighthouse. Greenit retrieves a score calculated with 3 metrics: the size of the DOM, the number of requests, and the size of the page (formula:(https://blog.octo.com/wp-content/uploads/2021/01/1-5-ecoindex-formule-recolored.jpg) Then, Google Lighthouse also provides a score calculated with 6 metrics such as the speed index, the time to interactive, etc. Those Chrome extensions, along with further data preprocessing, provided a score from 0-100, measuring the level of implementation of the 50 best practices recommended, which constitutes our dataset. As it is challenging for websites to implement all of the audit tools' suggested to improve the Eco-index score, we are interested in determining which factors have the greatest impact on the Eco-Index score. 

Thus, this research aims to first calculate which of the 50 practices have the most impact on the EcoIndex score. In order to uncover the feature importance that corresponds to the most essential practices to implement, four machine learning regression tasks are conducted, and two different models are used and evaluated for each of those: Random Forest and XGboost. Then, we generated a new dataset by applying the most important practices (manually increasing relevant scores), and we predict the new eco-index score, to be able to measure the impact of applying only those recommendations.

This study aims to provide simple and effective eco-concept recommendationa for websites and to aid developers. In addition, it is an expansion to the EcoIndex project (GreenIt.fr),  and an active contribution the EcoSonar (ecosonar.org) open-source project at Accenture Technology, which aims to build an application that provides targeted recommendations to websites. 



Metrics:

The score of the EcoIndex depends on the following metrics:

1. The Size of the DOM
2. Number of Requests
3. Size of the page

The score of Google Lighthouse Performance depends on the following metrics:

4. First Contentful Paint
5. Speed Index
6. Largest Contentful Paint
7. Time to Interactive
8. Total Blocking Time
9. Cumulative Layout Shift

Features:

List of Features from Green-IT Analysis :

1. cache header ratio
2. compress ratio
3. domains number
4. images resized in browser number
5. empty src tag number
6. inline style sheets number
7. inline js scripts number
8. error number
9. js validate
10. umber of requests
11. images downloaded not displayed number
12. max cookies length
13. percent minified css
14. percent minified js
15. total cookies size
16. redirect number
17. total min gains (bitmap images)
18. total size to optimize (svg images)
19. plugins number
20. print style sheets number
21. number social network button
22. style sheets number
23. eTags Ratio
24. total Fonts Size


List of Features from Google Ligthhouse:

25. viewport
26. serverResponsetime
27. mainthreadWorkBreakdown
28. bootupTime
29. fontDisplay
30. thirdPartySummary
31. thirdPartyFacades
32. lcpLazyLoaded
33. nonCompositedAnimations
34. domSize
35. usesLongCacheTtl
36. usesResponsiveImages
37. offscreenImages
38. unusedCssRules
39. unusedJavscript
40. usesOptimizedImages
41. modernImageFormats
42. usesTextCompression
43. usesHttp2
44. efficientAnimatedContent
45. legacyJavascript
46. totalByteWeight
47. noDocumentWrite
48. layoutShiftElements
49. usesPassiveEventListeners



