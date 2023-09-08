

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
orientation(spgt_0,156,73). coords(spgt_0,57,159,73).  visible(spgt_0,73). orientation(spgt_1,160,73). coords(spgt_1,57,149,73).  visible(spgt_1,73).
happensAt(disappear(spgt_0),72). happensAt(disappear(spgt_1),72).
1{simpleEvent(T,spgt_0,active); simpleEvent(T,spgt_0,inactive); simpleEvent(T,spgt_0,walking); simpleEvent(T,spgt_0,running)}1 :- T=48..72.
1{simpleEvent(T,spgt_1,active); simpleEvent(T,spgt_1,inactive); simpleEvent(T,spgt_1,walking); simpleEvent(T,spgt_1,running)}1 :- T=48..72.
:- holdsAt(moving(spgt_0,spgt_1),48). :- holdsAt(moving(spgt_1,spgt_0),48).
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
:- holdsAt(moving(spgt_0,spgt_1),73). :- holdsAt(moving(spgt_1,spgt_0),73).
:- simpleEvent(T,P,inactive), T>48 , T<72.
:- simpleEvent(T,P,running), T>48 , T<72.

#show simpleEvent/3.