

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

	person(lb1gt_1;lb1gt_2).
orientation(lb1gt_1,147,72). coords(lb1gt_1,74,104,72).  visible(lb1gt_1,72). orientation(lb1gt_2,142,72). coords(lb1gt_2,87,81,72).  visible(lb1gt_2,72).
orientation(lb1gt_1,147,73). coords(lb1gt_1,75,104,73).  visible(lb1gt_1,73). orientation(lb1gt_2,142,73). coords(lb1gt_2,87,81,73).  visible(lb1gt_2,73).
orientation(lb1gt_1,147,74). coords(lb1gt_1,75,105,74).  visible(lb1gt_1,74). orientation(lb1gt_2,142,74). coords(lb1gt_2,87,81,74).  visible(lb1gt_2,74).
orientation(lb1gt_1,147,75). coords(lb1gt_1,75,105,75).  visible(lb1gt_1,75). orientation(lb1gt_2,142,75). coords(lb1gt_2,88,81,75).  visible(lb1gt_2,75).
orientation(lb1gt_1,147,76). coords(lb1gt_1,75,106,76).  visible(lb1gt_1,76). orientation(lb1gt_2,142,76). coords(lb1gt_2,88,81,76).  visible(lb1gt_2,76).
orientation(lb1gt_1,147,77). coords(lb1gt_1,75,107,77).  visible(lb1gt_1,77). orientation(lb1gt_2,142,77). coords(lb1gt_2,88,81,77).  visible(lb1gt_2,77).
orientation(lb1gt_1,147,78). coords(lb1gt_1,75,107,78).  visible(lb1gt_1,78). orientation(lb1gt_2,142,78). coords(lb1gt_2,88,81,78).  visible(lb1gt_2,78).
orientation(lb1gt_1,147,79). coords(lb1gt_1,75,108,79).  visible(lb1gt_1,79). orientation(lb1gt_2,142,79). coords(lb1gt_2,88,81,79).  visible(lb1gt_2,79).
orientation(lb1gt_1,147,80). coords(lb1gt_1,75,109,80).  visible(lb1gt_1,80). orientation(lb1gt_2,142,80). coords(lb1gt_2,88,82,80).  visible(lb1gt_2,80).
orientation(lb1gt_1,147,81). coords(lb1gt_1,75,109,81).  visible(lb1gt_1,81). orientation(lb1gt_2,142,81). coords(lb1gt_2,88,82,81).  visible(lb1gt_2,81).
orientation(lb1gt_1,147,82). coords(lb1gt_1,75,109,82).  visible(lb1gt_1,82). orientation(lb1gt_2,142,82). coords(lb1gt_2,88,83,82).  visible(lb1gt_2,82).
orientation(lb1gt_1,147,83). coords(lb1gt_1,75,109,83).  visible(lb1gt_1,83). orientation(lb1gt_2,142,83). coords(lb1gt_2,88,83,83).  visible(lb1gt_2,83).
orientation(lb1gt_1,147,84). coords(lb1gt_1,75,109,84).  visible(lb1gt_1,84). orientation(lb1gt_2,142,84). coords(lb1gt_2,88,84,84).  visible(lb1gt_2,84).
orientation(lb1gt_1,147,85). coords(lb1gt_1,75,109,85).  visible(lb1gt_1,85). orientation(lb1gt_2,142,85). coords(lb1gt_2,88,85,85).  visible(lb1gt_2,85).
orientation(lb1gt_1,147,86). coords(lb1gt_1,75,109,86).  visible(lb1gt_1,86). orientation(lb1gt_2,142,86). coords(lb1gt_2,88,85,86).  visible(lb1gt_2,86).
orientation(lb1gt_1,147,87). coords(lb1gt_1,75,110,87).  visible(lb1gt_1,87). orientation(lb1gt_2,142,87). coords(lb1gt_2,88,86,87).  visible(lb1gt_2,87).
orientation(lb1gt_1,147,88). coords(lb1gt_1,75,110,88).  visible(lb1gt_1,88). orientation(lb1gt_2,142,88). coords(lb1gt_2,88,86,88).  visible(lb1gt_2,88).
orientation(lb1gt_1,147,89). coords(lb1gt_1,76,111,89).  visible(lb1gt_1,89). orientation(lb1gt_2,142,89). coords(lb1gt_2,88,86,89).  visible(lb1gt_2,89).
orientation(lb1gt_1,147,90). coords(lb1gt_1,76,113,90).  visible(lb1gt_1,90). orientation(lb1gt_2,142,90). coords(lb1gt_2,89,87,90).  visible(lb1gt_2,90).
orientation(lb1gt_1,147,91). coords(lb1gt_1,76,114,91).  visible(lb1gt_1,91). orientation(lb1gt_2,142,91). coords(lb1gt_2,89,87,91).  visible(lb1gt_2,91).
orientation(lb1gt_1,147,92). coords(lb1gt_1,76,115,92).  visible(lb1gt_1,92). orientation(lb1gt_2,142,92). coords(lb1gt_2,89,87,92).  visible(lb1gt_2,92).
orientation(lb1gt_1,147,93). coords(lb1gt_1,76,116,93).  visible(lb1gt_1,93). orientation(lb1gt_2,145,93). coords(lb1gt_2,89,87,93).  visible(lb1gt_2,93).
orientation(lb1gt_1,148,94). coords(lb1gt_1,77,116,94).  visible(lb1gt_1,94). orientation(lb1gt_2,145,94). coords(lb1gt_2,89,87,94).  visible(lb1gt_2,94).
orientation(lb1gt_1,148,95). coords(lb1gt_1,77,116,95).  visible(lb1gt_1,95). orientation(lb1gt_2,144,95). coords(lb1gt_2,89,87,95).  visible(lb1gt_2,95).
orientation(lb1gt_1,148,96). coords(lb1gt_1,77,117,96).  visible(lb1gt_1,96). orientation(lb1gt_2,145,96). coords(lb1gt_2,89,87,96).  visible(lb1gt_2,96).
orientation(lb1gt_1,148,97). coords(lb1gt_1,77,117,97).  visible(lb1gt_1,97). orientation(lb1gt_2,145,97). coords(lb1gt_2,89,87,97).  visible(lb1gt_2,97).
happensAt(disappear(lb1gt_1),96). happensAt(disappear(lb1gt_2),96).
1{simpleEvent(T,lb1gt_1,active); simpleEvent(T,lb1gt_1,inactive); simpleEvent(T,lb1gt_1,walking); simpleEvent(T,lb1gt_1,running)}1 :- T=72..96.
1{simpleEvent(T,lb1gt_2,active); simpleEvent(T,lb1gt_2,inactive); simpleEvent(T,lb1gt_2,walking); simpleEvent(T,lb1gt_2,running)}1 :- T=72..96.
:- holdsAt(moving(lb1gt_1,lb1gt_2),72). :- holdsAt(moving(lb1gt_2,lb1gt_1),72).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),73). :- not holdsAt(moving(lb1gt_2,lb1gt_1),73).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),74). :- not holdsAt(moving(lb1gt_2,lb1gt_1),74).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),75). :- not holdsAt(moving(lb1gt_2,lb1gt_1),75).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),76). :- not holdsAt(moving(lb1gt_2,lb1gt_1),76).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),77). :- not holdsAt(moving(lb1gt_2,lb1gt_1),77).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),78). :- not holdsAt(moving(lb1gt_2,lb1gt_1),78).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),79). :- not holdsAt(moving(lb1gt_2,lb1gt_1),79).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),80). :- not holdsAt(moving(lb1gt_2,lb1gt_1),80).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),81). :- not holdsAt(moving(lb1gt_2,lb1gt_1),81).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),82). :- not holdsAt(moving(lb1gt_2,lb1gt_1),82).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),83). :- not holdsAt(moving(lb1gt_2,lb1gt_1),83).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),84). :- not holdsAt(moving(lb1gt_2,lb1gt_1),84).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),85). :- not holdsAt(moving(lb1gt_2,lb1gt_1),85).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),86). :- not holdsAt(moving(lb1gt_2,lb1gt_1),86).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),87). :- not holdsAt(moving(lb1gt_2,lb1gt_1),87).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),88). :- not holdsAt(moving(lb1gt_2,lb1gt_1),88).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),89). :- not holdsAt(moving(lb1gt_2,lb1gt_1),89).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),90). :- not holdsAt(moving(lb1gt_2,lb1gt_1),90).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),91). :- not holdsAt(moving(lb1gt_2,lb1gt_1),91).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),92). :- not holdsAt(moving(lb1gt_2,lb1gt_1),92).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),93). :- not holdsAt(moving(lb1gt_2,lb1gt_1),93).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),94). :- not holdsAt(moving(lb1gt_2,lb1gt_1),94).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),95). :- not holdsAt(moving(lb1gt_2,lb1gt_1),95).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),96). :- not holdsAt(moving(lb1gt_2,lb1gt_1),96).
:- holdsAt(moving(lb1gt_1,lb1gt_2),97). :- holdsAt(moving(lb1gt_2,lb1gt_1),97).
:- simpleEvent(T,P,inactive), T>72 , T<96.
:- simpleEvent(T,P,running), T>72 , T<96.

#show simpleEvent/3.