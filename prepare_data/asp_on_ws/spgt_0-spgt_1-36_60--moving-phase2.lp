

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
orientation(spgt_0,156,36). coords(spgt_0,45,166,36).  visible(spgt_0,36). orientation(spgt_1,160,36). coords(spgt_1,45,157,36).  visible(spgt_1,36).
orientation(spgt_0,156,37). coords(spgt_0,45,166,37).  visible(spgt_0,37). orientation(spgt_1,160,37). coords(spgt_1,45,157,37).  visible(spgt_1,37).
orientation(spgt_0,156,38). coords(spgt_0,45,166,38).  visible(spgt_0,38). orientation(spgt_1,160,38). coords(spgt_1,45,157,38).  visible(spgt_1,38).
orientation(spgt_0,156,39). coords(spgt_0,45,166,39).  visible(spgt_0,39). orientation(spgt_1,160,39). coords(spgt_1,46,157,39).  visible(spgt_1,39).
orientation(spgt_0,156,40). coords(spgt_0,45,166,40).  visible(spgt_0,40). orientation(spgt_1,160,40). coords(spgt_1,46,157,40).  visible(spgt_1,40).
orientation(spgt_0,156,41). coords(spgt_0,46,166,41).  visible(spgt_0,41). orientation(spgt_1,160,41). coords(spgt_1,47,157,41).  visible(spgt_1,41).
orientation(spgt_0,156,42). coords(spgt_0,46,166,42).  visible(spgt_0,42). orientation(spgt_1,160,42). coords(spgt_1,47,157,42).  visible(spgt_1,42).
orientation(spgt_0,156,43). coords(spgt_0,46,165,43).  visible(spgt_0,43). orientation(spgt_1,160,43). coords(spgt_1,47,157,43).  visible(spgt_1,43).
orientation(spgt_0,156,44). coords(spgt_0,46,165,44).  visible(spgt_0,44). orientation(spgt_1,160,44). coords(spgt_1,47,156,44).  visible(spgt_1,44).
orientation(spgt_0,156,45). coords(spgt_0,46,165,45).  visible(spgt_0,45). orientation(spgt_1,160,45). coords(spgt_1,47,156,45).  visible(spgt_1,45).
orientation(spgt_0,156,46). coords(spgt_0,46,165,46).  visible(spgt_0,46). orientation(spgt_1,160,46). coords(spgt_1,48,156,46).  visible(spgt_1,46).
orientation(spgt_0,156,47). coords(spgt_0,47,164,47).  visible(spgt_0,47). orientation(spgt_1,160,47). coords(spgt_1,48,156,47).  visible(spgt_1,47).
orientation(spgt_0,156,48). coords(spgt_0,49,164,48).  visible(spgt_0,48). orientation(spgt_1,160,48). coords(spgt_1,48,156,48).  visible(spgt_1,48).
orientation(spgt_0,156,49). coords(spgt_0,49,165,49).  visible(spgt_0,49). orientation(spgt_1,160,49). coords(spgt_1,49,156,49).  visible(spgt_1,49).
orientation(spgt_0,156,50). coords(spgt_0,49,165,50).  visible(spgt_0,50). orientation(spgt_1,160,50). coords(spgt_1,49,156,50).  visible(spgt_1,50).
orientation(spgt_0,156,51). coords(spgt_0,50,165,51).  visible(spgt_0,51). orientation(spgt_1,160,51). coords(spgt_1,50,156,51).  visible(spgt_1,51).
orientation(spgt_0,156,52). coords(spgt_0,50,165,52).  visible(spgt_0,52). orientation(spgt_1,160,52). coords(spgt_1,51,156,52).  visible(spgt_1,52).
orientation(spgt_0,156,53). coords(spgt_0,51,164,53).  visible(spgt_0,53). orientation(spgt_1,160,53). coords(spgt_1,51,156,53).  visible(spgt_1,53).
orientation(spgt_0,156,54). coords(spgt_0,51,164,54).  visible(spgt_0,54). orientation(spgt_1,160,54). coords(spgt_1,52,155,54).  visible(spgt_1,54).
orientation(spgt_0,156,55). coords(spgt_0,52,164,55).  visible(spgt_0,55). orientation(spgt_1,160,55). coords(spgt_1,52,155,55).  visible(spgt_1,55).
orientation(spgt_0,156,56). coords(spgt_0,52,164,56).  visible(spgt_0,56). orientation(spgt_1,160,56). coords(spgt_1,53,155,56).  visible(spgt_1,56).
orientation(spgt_0,156,57). coords(spgt_0,52,164,57).  visible(spgt_0,57). orientation(spgt_1,160,57). coords(spgt_1,53,154,57).  visible(spgt_1,57).
orientation(spgt_0,156,58). coords(spgt_0,52,163,58).  visible(spgt_0,58). orientation(spgt_1,160,58). coords(spgt_1,53,154,58).  visible(spgt_1,58).
orientation(spgt_0,156,59). coords(spgt_0,52,163,59).  visible(spgt_0,59). orientation(spgt_1,160,59). coords(spgt_1,53,153,59).  visible(spgt_1,59).
orientation(spgt_0,156,60). coords(spgt_0,52,163,60).  visible(spgt_0,60). orientation(spgt_1,160,60). coords(spgt_1,54,153,60).  visible(spgt_1,60).
orientation(spgt_0,156,61). coords(spgt_0,52,163,61).  visible(spgt_0,61). orientation(spgt_1,160,61). coords(spgt_1,54,153,61).  visible(spgt_1,61).
happensAt(disappear(spgt_0),60). happensAt(disappear(spgt_1),60).
1{simpleEvent(T,spgt_0,active); simpleEvent(T,spgt_0,inactive); simpleEvent(T,spgt_0,walking); simpleEvent(T,spgt_0,running)}1 :- T=36..60.
1{simpleEvent(T,spgt_1,active); simpleEvent(T,spgt_1,inactive); simpleEvent(T,spgt_1,walking); simpleEvent(T,spgt_1,running)}1 :- T=36..60.
:- holdsAt(moving(spgt_0,spgt_1),36). :- holdsAt(moving(spgt_1,spgt_0),36).
:- not holdsAt(moving(spgt_0,spgt_1),37). :- not holdsAt(moving(spgt_1,spgt_0),37).
:- not holdsAt(moving(spgt_0,spgt_1),38). :- not holdsAt(moving(spgt_1,spgt_0),38).
:- not holdsAt(moving(spgt_0,spgt_1),39). :- not holdsAt(moving(spgt_1,spgt_0),39).
:- not holdsAt(moving(spgt_0,spgt_1),40). :- not holdsAt(moving(spgt_1,spgt_0),40).
:- not holdsAt(moving(spgt_0,spgt_1),41). :- not holdsAt(moving(spgt_1,spgt_0),41).
:- not holdsAt(moving(spgt_0,spgt_1),42). :- not holdsAt(moving(spgt_1,spgt_0),42).
:- not holdsAt(moving(spgt_0,spgt_1),43). :- not holdsAt(moving(spgt_1,spgt_0),43).
:- not holdsAt(moving(spgt_0,spgt_1),44). :- not holdsAt(moving(spgt_1,spgt_0),44).
:- not holdsAt(moving(spgt_0,spgt_1),45). :- not holdsAt(moving(spgt_1,spgt_0),45).
:- not holdsAt(moving(spgt_0,spgt_1),46). :- not holdsAt(moving(spgt_1,spgt_0),46).
:- not holdsAt(moving(spgt_0,spgt_1),47). :- not holdsAt(moving(spgt_1,spgt_0),47).
:- not holdsAt(moving(spgt_0,spgt_1),48). :- not holdsAt(moving(spgt_1,spgt_0),48).
:- not holdsAt(moving(spgt_0,spgt_1),49). :- not holdsAt(moving(spgt_1,spgt_0),49).
:- not holdsAt(moving(spgt_0,spgt_1),50). :- not holdsAt(moving(spgt_1,spgt_0),50).
:- not holdsAt(moving(spgt_0,spgt_1),51). :- not holdsAt(moving(spgt_1,spgt_0),51).
:- not holdsAt(moving(spgt_0,spgt_1),52). :- not holdsAt(moving(spgt_1,spgt_0),52).
:- not holdsAt(moving(spgt_0,spgt_1),53). :- not holdsAt(moving(spgt_1,spgt_0),53).
:- not holdsAt(moving(spgt_0,spgt_1),54). :- not holdsAt(moving(spgt_1,spgt_0),54).
:- not holdsAt(moving(spgt_0,spgt_1),55). :- not holdsAt(moving(spgt_1,spgt_0),55).
:- not holdsAt(moving(spgt_0,spgt_1),56). :- not holdsAt(moving(spgt_1,spgt_0),56).
:- not holdsAt(moving(spgt_0,spgt_1),57). :- not holdsAt(moving(spgt_1,spgt_0),57).
:- not holdsAt(moving(spgt_0,spgt_1),58). :- not holdsAt(moving(spgt_1,spgt_0),58).
:- not holdsAt(moving(spgt_0,spgt_1),59). :- not holdsAt(moving(spgt_1,spgt_0),59).
:- not holdsAt(moving(spgt_0,spgt_1),60). :- not holdsAt(moving(spgt_1,spgt_0),60).
:- holdsAt(moving(spgt_0,spgt_1),61). :- holdsAt(moving(spgt_1,spgt_0),61).
:- simpleEvent(T,P,inactive), T>36 , T<60.
:- simpleEvent(T,P,running), T>36 , T<60.

#show simpleEvent/3.