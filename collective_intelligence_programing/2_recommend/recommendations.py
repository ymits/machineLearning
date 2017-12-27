critics={
  'Lisa Rose': {
    'Lady in the Water': 2.5,
    'Snakes on a Plane': 3.5,
    'Just My Luck': 3.0,
    'Superman Returns': 3.5,
    'You, Me and Dupree': 2.5,
    'The Night Listener': 3.0
  },
  'Gene Seymour': {
    'Lady in the Water': 3.0,
    'Snakes on a Plane': 3.5,
    'Just My Luck': 1.5,
    'Superman Returns': 5.0,
    'You, Me and Dupree': 3.5,
    'The Night Listener': 3.0
  },
  'Michael Phillips': {
    'Lady in the Water': 2.5,
    'Snakes on a Plane': 3.0,
    'Superman Returns': 3.5,
    'The Night Listener': 4.0
  },
  'Claudia Puig': {
    'Snakes on a Plane': 3.5,
    'Just My Luck': 3.0,
    'Superman Returns': 4.0,
    'You, Me and Dupree': 2.5,
    'The Night Listener': 4.5
  },
  'Mick LaSalle': {
    'Lady in the Water': 3.0,
    'Snakes on a Plane': 4.0,
    'Just My Luck': 2.0,
    'Superman Returns': 3.0,
    'You, Me and Dupree': 2.0,
    'The Night Listener': 3.0
  },
  'Jack Matthews': {
    'Lady in the Water': 3.0,
    'Snakes on a Plane': 4.0,
    'Superman Returns': 5.0,
    'You, Me and Dupree': 3.5,
    'The Night Listener': 3.0
  },
  'Toby': {
    'Snakes on a Plane': 4.5,
    'Superman Returns': 4.0,
    'You, Me and Dupree': 1.0
  }
}

from math import sqrt
def sim_distance(prefs, person1, person2):
    # 二人共評価しているアイテムのリストを得る
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = prefs[person1][item] - prefs[person2][item]
    
    if len(si) == 0:
        return 0
    
    print(sum([si[item]**2 for item in si]))
    sum_of_squares = sum([si[item]**2 for item in si])
    print(sum_of_squares)
    
    return 1 / (1 + sqrt(sum_of_squares))

def sim_pearson(prefs, person1, person2):
    # 二人共評価しているアイテムのリストを得る
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1
            
    n = len(si)
    if n == 0:
        return 0
    
    avg1 = sum([prefs[person1][item] for item in si]) / n
    avg2 = sum([prefs[person2][item] for item in si]) / n
    covariance = sum([(prefs[person1][item] - avg1) * (prefs[person2][item] - avg2) for item in si]) / n
    stdev1 = sqrt(sum([(prefs[person1][item] - avg1) ** 2 for item in si]) / n)
    stdev2 = sqrt(sum([(prefs[person2][item] - avg2) ** 2 for item in si]) / n)
    
    return covariance / (stdev1 * stdev2)
    
def topMatches(prefs, person, n = 5, similarity = sim_pearson):
    scores = [(similarity(prefs, person, other), other) for other in prefs if person != other]
    
    scores.sort()
    scores.reverse()
    return scores[0:n]

def getRecommendations(prefs, person, similarity = sim_pearson):
    matches = topMatches(prefs, person)
    sims = {}
    for match in matches:
        sims[match[1]] = match[0]
        
    totals = {}
    simsum = {}

    for other in sims:
        for item in prefs[other]:
            if item not in prefs[person] or prefs[person][item] == 0:
                if sims[other] < 0 :
                    continue
                
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sims[other]

                simsum.setdefault(item, 0)
                simsum[item] += sims[other]
                
    rankings = [(totals[item] / simsum[item], item) for item in totals]

    rankings.sort()
    rankings.reverse()
    return rankings

def transformPrefs(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            result[item][person] = prefs[person][item]
    return result