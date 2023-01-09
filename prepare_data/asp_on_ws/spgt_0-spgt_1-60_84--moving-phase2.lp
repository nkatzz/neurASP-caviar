

holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).


happensAt(active(X),T) :- simpleEvent(T,X,active).
happensAt(inactive(X),T) :- simpleEvent(T,X,inactive).
happensAt(walking(X),T) :- simpleEvent(T,X,walking).
happensAt(running(X),T) :- simpleEvent(T,X,running). 


holdsByInertia(moving(X,Y),T+1) :- holdsAt(moving(X,Y),T+1), not initiatedAt(moving(X,Y),T).
holdsByInertia(meeting(X,Y),T+1) :- holdsAt(meeting(X,Y),T+1), not initiatedAt(meeting(X,Y),T).

:- holdsByInertia(moving(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2. 
:- holdsByInertia(meeting(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2.

	person(spgt_0;spgt_1).
orientation(spgt_0,156,60). coords(spgt_0,52,163,60).  visible(spgt_0,60). orientation(spgt_1,160,60). coords(spgt_1,54,153,60).  visible(spgt_1,60).
orientation(spgt_0,156,61). coords(spgt_0,53,163,61).  visible(spgt_0,61). orientation(spgt_1,160,61). coords(spgt_1,54,153,61).  visible(spgt_1,61).
orientation(spgt_0,156,62). coords(spgt_0,54,163,62).  visible(spgt_0,62). orientation(spgt_1,160,62). coords(spgt_1,54,152,62).  visible(spgt_1,62).
orientation(spgt_0,156,63). coords(spgt_0,55,163,63).  visible(spgt_0,63). orientation(spgt_1,160,63). coords(spgt_1,54,152,63).  visible(spgt_1,63).
orientation(spgt_0,156,64). coords(spgt_0,56,162,64).  visible(spgt_0,64). orientation(spgt_1,160,64). coords(spgt_1,55,151,64).  visible(spgt_1,64).
orientation(spgt_0,156,65). coords(spgt_0,56,161,65).  visible(spgt_0,65). orientation(spgt_1,160,65). coords(spgt_1,55,151,65).  visible(spgt_1,65).
orientation(spgt_0,156,66). coords(spgt_0,57,161,66).  visible(spgt_0,66). orientation(spgt_1,160,66). coords(spgt_1,56,151,66).  visible(spgt_1,66).
orientation(spgt_0,156,67). coords(spgt_0,57,161,67).  visible(spgt_0,67). orientation(spgt_1,160,67). coords(spgt_1,56,151,67).  visible(spgt_1,67).
orientation(spgt_0,156,68). coords(spgt_0,57,161,68).  visible(spgt_0,68). orientation(spgt_1,160,68). coords(spgt_1,56,151,68).  visible(spgt_1,68).
orientation(spgt_0,156,69). coords(spgt_0,57,160,69).  visible(spgt_0,69). orientation(spgt_1,160,69). coords(spgt_1,56,151,69).  visible(spgt_1,69).
orientation(spgt_0,156,70). coords(spgt_0,57,160,70).  visible(spgt_0,70). orientation(spgt_1,160,70). coords(spgt_1,56,150,70).  visible(spgt_1,70).
orientation(spgt_0,156,71). coords(spgt_0,57,159,71).  visible(spgt_0,71). orientation(spgt_1,160,71). coords(spgt_1,56,150,71).  visible(spgt_1,71).
orientation(spgt_0,156,72). coords(spgt_0,57,159,72).  visible(spgt_0,72). orientation(spgt_1,160,72). coords(spgt_1,57,149,72).  visible(spgt_1,72).
orientation(spgt_0,156,73). coords(spgt_0,58,159,73).  visible(spgt_0,73). orientation(spgt_1,160,73). coords(spgt_1,58,149,73).  visible(spgt_1,73).
orientation(spgt_0,156,74). coords(spgt_0,59,159,74).  visible(spgt_0,74). orientation(spgt_1,160,74). coords(spgt_1,59,149,74).  visible(spgt_1,74).
orientation(spgt_0,156,75). coords(spgt_0,60,159,75).  visible(spgt_0,75). orientation(spgt_1,160,75). coords(spgt_1,60,148,75).  visible(spgt_1,75).
orientation(spgt_0,156,76). coords(spgt_0,61,158,76).  visible(spgt_0,76). orientation(spgt_1,160,76). coords(spgt_1,60,148,76).  visible(spgt_1,76).
orientation(spgt_0,156,77). coords(spgt_0,61,158,77).  visible(spgt_0,77). orientation(spgt_1,160,77). coords(spgt_1,60,148,77).  visible(spgt_1,77).
orientation(spgt_0,156,78). coords(spgt_0,62,158,78).  visible(spgt_0,78). orientation(spgt_1,160,78). coords(spgt_1,61,147,78).  visible(spgt_1,78).
orientation(spgt_0,156,79). coords(spgt_0,62,157,79).  visible(spgt_0,79). orientation(spgt_1,160,79). coords(spgt_1,63,147,79).  visible(spgt_1,79).
orientation(spgt_0,156,80). coords(spgt_0,63,157,80).  visible(spgt_0,80). orientation(spgt_1,160,80). coords(spgt_1,63,147,80).  visible(spgt_1,80).
orientation(spgt_0,156,81). coords(spgt_0,63,157,81).  visible(spgt_0,81). orientation(spgt_1,160,81). coords(spgt_1,63,146,81).  visible(spgt_1,81).
orientation(spgt_0,156,82). coords(spgt_0,63,157,82).  visible(spgt_0,82). orientation(spgt_1,160,82). coords(spgt_1,63,146,82).  visible(spgt_1,82).
orientation(spgt_0,152,83). coords(spgt_0,64,156,83).  visible(spgt_0,83). orientation(spgt_1,160,83). coords(spgt_1,64,145,83).  visible(spgt_1,83).
orientation(spgt_0,152,84). coords(spgt_0,64,155,84).  visible(spgt_0,84). orientation(spgt_1,160,84). coords(spgt_1,64,145,84).  visible(spgt_1,84).
orientation(spgt_0,152,85). coords(spgt_0,64,155,85).  visible(spgt_0,85). orientation(spgt_1,160,85). coords(spgt_1,64,145,85).  visible(spgt_1,85).
happensAt(disappear(spgt_0),84). happensAt(disappear(spgt_1),84).
1{simpleEvent(T,spgt_0,active); simpleEvent(T,spgt_0,inactive); simpleEvent(T,spgt_0,walking); simpleEvent(T,spgt_0,running)}1 :- T=60..84.
1{simpleEvent(T,spgt_1,active); simpleEvent(T,spgt_1,inactive); simpleEvent(T,spgt_1,walking); simpleEvent(T,spgt_1,running)}1 :- T=60..84.
:- holdsAt(moving(spgt_0,spgt_1),60). :- holdsAt(moving(spgt_1,spgt_0),60).
:- not holdsAt(moving(spgt_0,spgt_1),61). :- not holdsAt(moving(spgt_1,spgt_0),61).
:- not holdsAt(moving(spgt_0,spgt_1),62). :- not holdsAt(moving(spgt_1,spgt_0),62).
:- not holdsAt(moving(spgt_0,spgt_1),63). :- not holdsAt(moving(spgt_1,spgt_0),63).
:- not holdsAt(moving(spgt_0,spgt_1),64). :- not holdsAt(moving(spgt_1,spgt_0),64).
:- not holdsAt(moving(spgt_0,spgt_1),65). :- not holdsAt(moving(spgt_1,spgt_0),65).
:- not holdsAt(moving(spgt_0,spgt_1),66). :- not holdsAt(moving(spgt_1,spgt_0),66).
:- not holdsAt(moving(spgt_0,spgt_1),67). :- not holdsAt(moving(spgt_1,spgt_0),67).
:- not holdsAt(moving(spgt_0,spgt_1),68). :- not holdsAt(moving(spgt_1,spgt_0),68).
:- not holdsAt(moving(spgt_0,spgt_1),69). :- not holdsAt(moving(spgt_1,spgt_0),69).
:- not holdsAt(moving(spgt_0,spgt_1),70). :- not holdsAt(moving(spgt_1,spgt_0),70).
:- not holdsAt(moving(spgt_0,spgt_1),71). :- not holdsAt(moving(spgt_1,spgt_0),71).
:- not holdsAt(moving(spgt_0,spgt_1),72). :- not holdsAt(moving(spgt_1,spgt_0),72).
:- not holdsAt(moving(spgt_0,spgt_1),73). :- not holdsAt(moving(spgt_1,spgt_0),73).
:- not holdsAt(moving(spgt_0,spgt_1),74). :- not holdsAt(moving(spgt_1,spgt_0),74).
:- not holdsAt(moving(spgt_0,spgt_1),75). :- not holdsAt(moving(spgt_1,spgt_0),75).
:- not holdsAt(moving(spgt_0,spgt_1),76). :- not holdsAt(moving(spgt_1,spgt_0),76).
:- not holdsAt(moving(spgt_0,spgt_1),77). :- not holdsAt(moving(spgt_1,spgt_0),77).
:- not holdsAt(moving(spgt_0,spgt_1),78). :- not holdsAt(moving(spgt_1,spgt_0),78).
:- not holdsAt(moving(spgt_0,spgt_1),79). :- not holdsAt(moving(spgt_1,spgt_0),79).
:- not holdsAt(moving(spgt_0,spgt_1),80). :- not holdsAt(moving(spgt_1,spgt_0),80).
:- not holdsAt(moving(spgt_0,spgt_1),81). :- not holdsAt(moving(spgt_1,spgt_0),81).
:- not holdsAt(moving(spgt_0,spgt_1),82). :- not holdsAt(moving(spgt_1,spgt_0),82).
:- not holdsAt(moving(spgt_0,spgt_1),83). :- not holdsAt(moving(spgt_1,spgt_0),83).
:- not holdsAt(moving(spgt_0,spgt_1),84). :- not holdsAt(moving(spgt_1,spgt_0),84).
:- holdsAt(moving(spgt_0,spgt_1),85). :- holdsAt(moving(spgt_1,spgt_0),85).
:- simpleEvent(T,P,inactive), T>60 , T<84.
:- simpleEvent(T,P,running), T>60 , T<84.

#show simpleEvent/3.