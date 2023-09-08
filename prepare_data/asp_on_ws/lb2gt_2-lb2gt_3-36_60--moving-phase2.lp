

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

	person(lb2gt_2;lb2gt_3).
orientation(lb2gt_2,147,36). coords(lb2gt_2,20,53,36).  visible(lb2gt_2,36). orientation(lb2gt_3,146,36). coords(lb2gt_3,23,40,36).  visible(lb2gt_3,36).
orientation(lb2gt_2,147,37). coords(lb2gt_2,20,53,37).  visible(lb2gt_2,37). orientation(lb2gt_3,146,37). coords(lb2gt_3,23,40,37).  visible(lb2gt_3,37).
orientation(lb2gt_2,147,38). coords(lb2gt_2,20,54,38).  visible(lb2gt_2,38). orientation(lb2gt_3,146,38). coords(lb2gt_3,21,39,38).  visible(lb2gt_3,38).
orientation(lb2gt_2,147,39). coords(lb2gt_2,20,53,39).  visible(lb2gt_2,39). orientation(lb2gt_3,146,39). coords(lb2gt_3,21,40,39).  visible(lb2gt_3,39).
orientation(lb2gt_2,147,40). coords(lb2gt_2,20,53,40).  visible(lb2gt_2,40). orientation(lb2gt_3,146,40). coords(lb2gt_3,20,40,40).  visible(lb2gt_3,40).
orientation(lb2gt_2,147,41). coords(lb2gt_2,20,53,41).  visible(lb2gt_2,41). orientation(lb2gt_3,146,41). coords(lb2gt_3,20,40,41).  visible(lb2gt_3,41).
orientation(lb2gt_2,147,42). coords(lb2gt_2,20,54,42).  visible(lb2gt_2,42). orientation(lb2gt_3,146,42). coords(lb2gt_3,20,40,42).  visible(lb2gt_3,42).
orientation(lb2gt_2,147,43). coords(lb2gt_2,19,54,43).  visible(lb2gt_2,43). orientation(lb2gt_3,146,43). coords(lb2gt_3,20,40,43).  visible(lb2gt_3,43).
orientation(lb2gt_2,147,44). coords(lb2gt_2,19,54,44).  visible(lb2gt_2,44). orientation(lb2gt_3,146,44). coords(lb2gt_3,20,40,44).  visible(lb2gt_3,44).
orientation(lb2gt_2,147,45). coords(lb2gt_2,19,54,45).  visible(lb2gt_2,45). orientation(lb2gt_3,146,45). coords(lb2gt_3,20,40,45).  visible(lb2gt_3,45).
orientation(lb2gt_2,147,46). coords(lb2gt_2,19,54,46).  visible(lb2gt_2,46). orientation(lb2gt_3,146,46). coords(lb2gt_3,20,40,46).  visible(lb2gt_3,46).
orientation(lb2gt_2,147,47). coords(lb2gt_2,19,54,47).  visible(lb2gt_2,47). orientation(lb2gt_3,146,47). coords(lb2gt_3,20,40,47).  visible(lb2gt_3,47).
orientation(lb2gt_2,147,48). coords(lb2gt_2,19,54,48).  visible(lb2gt_2,48). orientation(lb2gt_3,146,48). coords(lb2gt_3,19,40,48).  visible(lb2gt_3,48).
orientation(lb2gt_2,147,49). coords(lb2gt_2,19,54,49).  visible(lb2gt_2,49). orientation(lb2gt_3,146,49). coords(lb2gt_3,19,41,49).  visible(lb2gt_3,49).
orientation(lb2gt_2,147,50). coords(lb2gt_2,19,54,50).  visible(lb2gt_2,50). orientation(lb2gt_3,146,50). coords(lb2gt_3,19,42,50).  visible(lb2gt_3,50).
orientation(lb2gt_2,147,51). coords(lb2gt_2,19,54,51).  visible(lb2gt_2,51). orientation(lb2gt_3,146,51). coords(lb2gt_3,18,42,51).  visible(lb2gt_3,51).
orientation(lb2gt_2,147,52). coords(lb2gt_2,19,54,52).  visible(lb2gt_2,52). orientation(lb2gt_3,146,52). coords(lb2gt_3,18,43,52).  visible(lb2gt_3,52).
orientation(lb2gt_2,147,53). coords(lb2gt_2,19,54,53).  visible(lb2gt_2,53). orientation(lb2gt_3,146,53). coords(lb2gt_3,18,43,53).  visible(lb2gt_3,53).
orientation(lb2gt_2,147,54). coords(lb2gt_2,19,54,54).  visible(lb2gt_2,54). orientation(lb2gt_3,146,54). coords(lb2gt_3,17,44,54).  visible(lb2gt_3,54).
orientation(lb2gt_2,147,55). coords(lb2gt_2,19,54,55).  visible(lb2gt_2,55). orientation(lb2gt_3,146,55). coords(lb2gt_3,17,45,55).  visible(lb2gt_3,55).
orientation(lb2gt_2,147,56). coords(lb2gt_2,19,54,56).  visible(lb2gt_2,56). orientation(lb2gt_3,146,56). coords(lb2gt_3,17,45,56).  visible(lb2gt_3,56).
orientation(lb2gt_2,147,57). coords(lb2gt_2,19,54,57).  visible(lb2gt_2,57). orientation(lb2gt_3,146,57). coords(lb2gt_3,17,45,57).  visible(lb2gt_3,57).
orientation(lb2gt_2,147,58). coords(lb2gt_2,19,54,58).  visible(lb2gt_2,58). orientation(lb2gt_3,146,58). coords(lb2gt_3,17,45,58).  visible(lb2gt_3,58).
orientation(lb2gt_2,147,59). coords(lb2gt_2,19,54,59).  visible(lb2gt_2,59). orientation(lb2gt_3,146,59). coords(lb2gt_3,18,45,59).  visible(lb2gt_3,59).
orientation(lb2gt_2,147,60). coords(lb2gt_2,19,54,60).  happensAt(disappear(lb2gt_2),60). orientation(lb2gt_3,146,60). coords(lb2gt_3,18,45,60).  visible(lb2gt_3,60).
orientation(lb2gt_2,147,61). coords(lb2gt_2,19,54,61).  happensAt(disappear(lb2gt_2),61). orientation(lb2gt_3,146,61). coords(lb2gt_3,18,45,61).  visible(lb2gt_3,61).
happensAt(disappear(lb2gt_2),60). happensAt(disappear(lb2gt_3),60).
1{simpleEvent(T,lb2gt_2,active); simpleEvent(T,lb2gt_2,inactive); simpleEvent(T,lb2gt_2,walking); simpleEvent(T,lb2gt_2,running)}1 :- T=36..60.
1{simpleEvent(T,lb2gt_3,active); simpleEvent(T,lb2gt_3,inactive); simpleEvent(T,lb2gt_3,walking); simpleEvent(T,lb2gt_3,running)}1 :- T=36..60.
:- holdsAt(moving(lb2gt_2,lb2gt_3),36). :- holdsAt(moving(lb2gt_3,lb2gt_2),36).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),37). :- not holdsAt(moving(lb2gt_3,lb2gt_2),37).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),38). :- not holdsAt(moving(lb2gt_3,lb2gt_2),38).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),39). :- not holdsAt(moving(lb2gt_3,lb2gt_2),39).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),40). :- not holdsAt(moving(lb2gt_3,lb2gt_2),40).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),41). :- not holdsAt(moving(lb2gt_3,lb2gt_2),41).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),42). :- not holdsAt(moving(lb2gt_3,lb2gt_2),42).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),43). :- not holdsAt(moving(lb2gt_3,lb2gt_2),43).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),44). :- not holdsAt(moving(lb2gt_3,lb2gt_2),44).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),45). :- not holdsAt(moving(lb2gt_3,lb2gt_2),45).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),46). :- not holdsAt(moving(lb2gt_3,lb2gt_2),46).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),47). :- not holdsAt(moving(lb2gt_3,lb2gt_2),47).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),48). :- not holdsAt(moving(lb2gt_3,lb2gt_2),48).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),49). :- not holdsAt(moving(lb2gt_3,lb2gt_2),49).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),50). :- not holdsAt(moving(lb2gt_3,lb2gt_2),50).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),51). :- not holdsAt(moving(lb2gt_3,lb2gt_2),51).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),52). :- not holdsAt(moving(lb2gt_3,lb2gt_2),52).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),53). :- not holdsAt(moving(lb2gt_3,lb2gt_2),53).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),54). :- not holdsAt(moving(lb2gt_3,lb2gt_2),54).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),55). :- not holdsAt(moving(lb2gt_3,lb2gt_2),55).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),56). :- not holdsAt(moving(lb2gt_3,lb2gt_2),56).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),57). :- not holdsAt(moving(lb2gt_3,lb2gt_2),57).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),58). :- not holdsAt(moving(lb2gt_3,lb2gt_2),58).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),59). :- not holdsAt(moving(lb2gt_3,lb2gt_2),59).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),60). :- not holdsAt(moving(lb2gt_3,lb2gt_2),60).
:- holdsAt(moving(lb2gt_2,lb2gt_3),61). :- holdsAt(moving(lb2gt_3,lb2gt_2),61).
:- simpleEvent(T,P,inactive), T>36 , T<60.
:- simpleEvent(T,P,running), T>36 , T<60.

#show simpleEvent/3.