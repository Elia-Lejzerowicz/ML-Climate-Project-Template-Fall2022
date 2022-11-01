Today, the internet and the technologies supporting it account for 3.7% of total carbon emissions, which is equivalent to the entire airline industry. Moreover, this number is expected to double in the next 3 years. There is a need for change and impact. As a result, the goal of this research is to find a way to reduce the amount of carbon dioxide emitted from websites. Indeed, the more complex a website is, the more energy it will require to load and the greater its carbon emissions. There is a need to inform developers on what practices to implement in order to reduce their website’s carbon footprint. 

We generated a dataset by analyzing more than 500 websites using two open-source audit tools: GreenIT-analysis and Google Lighthouse. Greenit retrieves a score calculated with 3 metrics: the size of the DOM, the number of requests, and the size of the page (formula:(https://blog.octo.com/wp-content/uploads/2021/01/1-5-ecoindex-formule-recolored.jpg) Then, Google Lighthouse also provides a score calculated with 6 metrics such as the speed index, the time to interactive, etc. Those Chrome extensions, along with further data preprocessing, provided a score from 0-100, measuring the level of implementation of the 50 best practices from GreenIt and Google Ligthouse. As a result, there is a total of 9 metric and 50 practices. Those practices could have a great impact on the carbon footprint if better implemented. 

Thus, this research aims to first calculate which practices have the most impact on a specific metric (from the 9 metrics analyzed). This will be calculated by performing 9 classification problems, with the practices as dependent variables (x) and the metric as the independent variable (y) . We will be able to calculate the importance of the features and thus discover the most relevant practices to implement among the 50 practices for each individual website. Then, we will generate a new dataset by applying the most important practices (manually increasing relevant scoes), and we will predict the new 9 metrics along with the eco-index score. This will enable us to predict how much a website is able to reduce its carbon footprint by applying relevant practices.

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





