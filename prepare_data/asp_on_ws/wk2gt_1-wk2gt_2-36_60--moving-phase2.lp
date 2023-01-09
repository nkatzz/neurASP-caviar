

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

	person(wk2gt_1;wk2gt_2).
orientation(wk2gt_1,143,36). coords(wk2gt_1,109,58,36).  visible(wk2gt_1,36). orientation(wk2gt_2,141,36). coords(wk2gt_2,97,66,36).  visible(wk2gt_2,36).
orientation(wk2gt_1,143,37). coords(wk2gt_1,109,58,37).  visible(wk2gt_1,37). orientation(wk2gt_2,141,37). coords(wk2gt_2,97,66,37).  visible(wk2gt_2,37).
orientation(wk2gt_1,143,38). coords(wk2gt_1,109,57,38).  visible(wk2gt_1,38). orientation(wk2gt_2,141,38). coords(wk2gt_2,97,66,38).  visible(wk2gt_2,38).
orientation(wk2gt_1,143,39). coords(wk2gt_1,109,57,39).  visible(wk2gt_1,39). orientation(wk2gt_2,141,39). coords(wk2gt_2,97,65,39).  visible(wk2gt_2,39).
orientation(wk2gt_1,143,40). coords(wk2gt_1,109,56,40).  visible(wk2gt_1,40). orientation(wk2gt_2,141,40). coords(wk2gt_2,98,65,40).  visible(wk2gt_2,40).
orientation(wk2gt_1,143,41). coords(wk2gt_1,109,56,41).  visible(wk2gt_1,41). orientation(wk2gt_2,141,41). coords(wk2gt_2,98,64,41).  visible(wk2gt_2,41).
orientation(wk2gt_1,143,42). coords(wk2gt_1,109,57,42).  visible(wk2gt_1,42). orientation(wk2gt_2,141,42). coords(wk2gt_2,98,64,42).  visible(wk2gt_2,42).
orientation(wk2gt_1,143,43). coords(wk2gt_1,109,57,43).  visible(wk2gt_1,43). orientation(wk2gt_2,141,43). coords(wk2gt_2,98,64,43).  visible(wk2gt_2,43).
orientation(wk2gt_1,143,44). coords(wk2gt_1,109,57,44).  visible(wk2gt_1,44). orientation(wk2gt_2,141,44). coords(wk2gt_2,98,63,44).  visible(wk2gt_2,44).
orientation(wk2gt_1,143,45). coords(wk2gt_1,109,56,45).  visible(wk2gt_1,45). orientation(wk2gt_2,141,45). coords(wk2gt_2,98,63,45).  visible(wk2gt_2,45).
orientation(wk2gt_1,143,46). coords(wk2gt_1,109,56,46).  visible(wk2gt_1,46). orientation(wk2gt_2,141,46). coords(wk2gt_2,98,63,46).  visible(wk2gt_2,46).
orientation(wk2gt_1,143,47). coords(wk2gt_1,109,56,47).  visible(wk2gt_1,47). orientation(wk2gt_2,141,47). coords(wk2gt_2,98,63,47).  visible(wk2gt_2,47).
orientation(wk2gt_1,143,48). coords(wk2gt_1,109,56,48).  visible(wk2gt_1,48). orientation(wk2gt_2,141,48). coords(wk2gt_2,98,63,48).  visible(wk2gt_2,48).
orientation(wk2gt_1,143,49). coords(wk2gt_1,110,56,49).  visible(wk2gt_1,49). orientation(wk2gt_2,141,49). coords(wk2gt_2,98,63,49).  visible(wk2gt_2,49).
orientation(wk2gt_1,143,50). coords(wk2gt_1,110,56,50).  visible(wk2gt_1,50). orientation(wk2gt_2,141,50). coords(wk2gt_2,98,63,50).  visible(wk2gt_2,50).
orientation(wk2gt_1,143,51). coords(wk2gt_1,110,56,51).  visible(wk2gt_1,51). orientation(wk2gt_2,141,51). coords(wk2gt_2,98,63,51).  visible(wk2gt_2,51).
orientation(wk2gt_1,143,52). coords(wk2gt_1,110,56,52).  visible(wk2gt_1,52). orientation(wk2gt_2,141,52). coords(wk2gt_2,98,63,52).  visible(wk2gt_2,52).
orientation(wk2gt_1,143,53). coords(wk2gt_1,110,56,53).  visible(wk2gt_1,53). orientation(wk2gt_2,141,53). coords(wk2gt_2,98,63,53).  visible(wk2gt_2,53).
orientation(wk2gt_1,143,54). coords(wk2gt_1,110,56,54).  visible(wk2gt_1,54). orientation(wk2gt_2,141,54). coords(wk2gt_2,98,63,54).  visible(wk2gt_2,54).
orientation(wk2gt_1,143,55). coords(wk2gt_1,110,56,55).  visible(wk2gt_1,55). orientation(wk2gt_2,141,55). coords(wk2gt_2,98,63,55).  visible(wk2gt_2,55).
orientation(wk2gt_1,143,56). coords(wk2gt_1,110,56,56).  visible(wk2gt_1,56). orientation(wk2gt_2,141,56). coords(wk2gt_2,98,63,56).  visible(wk2gt_2,56).
orientation(wk2gt_1,143,57). coords(wk2gt_1,111,56,57).  visible(wk2gt_1,57). orientation(wk2gt_2,141,57). coords(wk2gt_2,98,63,57).  visible(wk2gt_2,57).
orientation(wk2gt_1,143,58). coords(wk2gt_1,112,56,58).  visible(wk2gt_1,58). orientation(wk2gt_2,141,58). coords(wk2gt_2,98,63,58).  visible(wk2gt_2,58).
orientation(wk2gt_1,143,59). coords(wk2gt_1,112,55,59).  visible(wk2gt_1,59). orientation(wk2gt_2,141,59). coords(wk2gt_2,98,63,59).  visible(wk2gt_2,59).
orientation(wk2gt_1,140,60). coords(wk2gt_1,112,55,60).  visible(wk2gt_1,60). orientation(wk2gt_2,141,60). coords(wk2gt_2,98,63,60).  visible(wk2gt_2,60).
orientation(wk2gt_1,140,61). coords(wk2gt_1,112,55,61).  visible(wk2gt_1,61). orientation(wk2gt_2,141,61). coords(wk2gt_2,98,63,61).  visible(wk2gt_2,61).
happensAt(disappear(wk2gt_1),60). happensAt(disappear(wk2gt_2),60).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=36..60.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=36..60.
:- holdsAt(moving(wk2gt_1,wk2gt_2),36). :- holdsAt(moving(wk2gt_2,wk2gt_1),36).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),37). :- not holdsAt(moving(wk2gt_2,wk2gt_1),37).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),38). :- not holdsAt(moving(wk2gt_2,wk2gt_1),38).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),39). :- not holdsAt(moving(wk2gt_2,wk2gt_1),39).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),40). :- not holdsAt(moving(wk2gt_2,wk2gt_1),40).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),41). :- not holdsAt(moving(wk2gt_2,wk2gt_1),41).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),42). :- not holdsAt(moving(wk2gt_2,wk2gt_1),42).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),43). :- not holdsAt(moving(wk2gt_2,wk2gt_1),43).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),44). :- not holdsAt(moving(wk2gt_2,wk2gt_1),44).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),45). :- not holdsAt(moving(wk2gt_2,wk2gt_1),45).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),46). :- not holdsAt(moving(wk2gt_2,wk2gt_1),46).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),47). :- not holdsAt(moving(wk2gt_2,wk2gt_1),47).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),48). :- not holdsAt(moving(wk2gt_2,wk2gt_1),48).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),49). :- not holdsAt(moving(wk2gt_2,wk2gt_1),49).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),50). :- not holdsAt(moving(wk2gt_2,wk2gt_1),50).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),51). :- not holdsAt(moving(wk2gt_2,wk2gt_1),51).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),52). :- not holdsAt(moving(wk2gt_2,wk2gt_1),52).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),53). :- not holdsAt(moving(wk2gt_2,wk2gt_1),53).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),54). :- not holdsAt(moving(wk2gt_2,wk2gt_1),54).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),55). :- not holdsAt(moving(wk2gt_2,wk2gt_1),55).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),56). :- not holdsAt(moving(wk2gt_2,wk2gt_1),56).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),57). :- not holdsAt(moving(wk2gt_2,wk2gt_1),57).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),58). :- not holdsAt(moving(wk2gt_2,wk2gt_1),58).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),59). :- not holdsAt(moving(wk2gt_2,wk2gt_1),59).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),60). :- not holdsAt(moving(wk2gt_2,wk2gt_1),60).
:- holdsAt(moving(wk2gt_1,wk2gt_2),61). :- holdsAt(moving(wk2gt_2,wk2gt_1),61).
:- simpleEvent(T,P,inactive), T>36 , T<60.
:- simpleEvent(T,P,running), T>36 , T<60.

#show simpleEvent/3.