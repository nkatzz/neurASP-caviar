

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

	person(mc1gt_0;mc1gt_1).
orientation(mc1gt_0,148,76). coords(mc1gt_0,79,96,76).  visible(mc1gt_0,76). orientation(mc1gt_1,145,76). coords(mc1gt_1,94,83,76).  visible(mc1gt_1,76).
orientation(mc1gt_0,148,77). coords(mc1gt_0,79,96,77).  visible(mc1gt_0,77). orientation(mc1gt_1,145,77). coords(mc1gt_1,94,83,77).  visible(mc1gt_1,77).
orientation(mc1gt_0,148,78). coords(mc1gt_0,79,96,78).  visible(mc1gt_0,78). orientation(mc1gt_1,145,78). coords(mc1gt_1,94,83,78).  visible(mc1gt_1,78).
orientation(mc1gt_0,148,79). coords(mc1gt_0,79,96,79).  visible(mc1gt_0,79). orientation(mc1gt_1,145,79). coords(mc1gt_1,94,83,79).  visible(mc1gt_1,79).
orientation(mc1gt_0,148,80). coords(mc1gt_0,79,96,80).  visible(mc1gt_0,80). orientation(mc1gt_1,145,80). coords(mc1gt_1,94,83,80).  visible(mc1gt_1,80).
orientation(mc1gt_0,148,81). coords(mc1gt_0,79,96,81).  visible(mc1gt_0,81). orientation(mc1gt_1,145,81). coords(mc1gt_1,94,83,81).  visible(mc1gt_1,81).
orientation(mc1gt_0,148,82). coords(mc1gt_0,79,96,82).  visible(mc1gt_0,82). orientation(mc1gt_1,145,82). coords(mc1gt_1,94,83,82).  visible(mc1gt_1,82).
orientation(mc1gt_0,148,83). coords(mc1gt_0,79,96,83).  visible(mc1gt_0,83). orientation(mc1gt_1,145,83). coords(mc1gt_1,94,83,83).  visible(mc1gt_1,83).
orientation(mc1gt_0,148,84). coords(mc1gt_0,79,96,84).  visible(mc1gt_0,84). orientation(mc1gt_1,145,84). coords(mc1gt_1,94,83,84).  visible(mc1gt_1,84).
orientation(mc1gt_0,148,85). coords(mc1gt_0,79,96,85).  visible(mc1gt_0,85). orientation(mc1gt_1,145,85). coords(mc1gt_1,94,83,85).  visible(mc1gt_1,85).
orientation(mc1gt_0,148,86). coords(mc1gt_0,79,96,86).  visible(mc1gt_0,86). orientation(mc1gt_1,145,86). coords(mc1gt_1,94,83,86).  visible(mc1gt_1,86).
orientation(mc1gt_0,148,87). coords(mc1gt_0,79,96,87).  visible(mc1gt_0,87). orientation(mc1gt_1,145,87). coords(mc1gt_1,94,83,87).  visible(mc1gt_1,87).
orientation(mc1gt_0,148,88). coords(mc1gt_0,79,96,88).  visible(mc1gt_0,88). orientation(mc1gt_1,145,88). coords(mc1gt_1,94,83,88).  visible(mc1gt_1,88).
orientation(mc1gt_0,148,89). coords(mc1gt_0,79,96,89).  visible(mc1gt_0,89). orientation(mc1gt_1,145,89). coords(mc1gt_1,94,83,89).  visible(mc1gt_1,89).
orientation(mc1gt_0,148,90). coords(mc1gt_0,79,96,90).  visible(mc1gt_0,90). orientation(mc1gt_1,145,90). coords(mc1gt_1,94,83,90).  visible(mc1gt_1,90).
orientation(mc1gt_0,148,91). coords(mc1gt_0,79,96,91).  visible(mc1gt_0,91). orientation(mc1gt_1,145,91). coords(mc1gt_1,94,83,91).  visible(mc1gt_1,91).
orientation(mc1gt_0,148,92). coords(mc1gt_0,79,96,92).  visible(mc1gt_0,92). orientation(mc1gt_1,145,92). coords(mc1gt_1,94,83,92).  visible(mc1gt_1,92).
orientation(mc1gt_0,148,93). coords(mc1gt_0,79,96,93).  visible(mc1gt_0,93). orientation(mc1gt_1,145,93). coords(mc1gt_1,94,83,93).  visible(mc1gt_1,93).
orientation(mc1gt_0,148,94). coords(mc1gt_0,79,96,94).  visible(mc1gt_0,94). orientation(mc1gt_1,145,94). coords(mc1gt_1,94,83,94).  visible(mc1gt_1,94).
orientation(mc1gt_0,148,95). coords(mc1gt_0,79,96,95).  visible(mc1gt_0,95). orientation(mc1gt_1,145,95). coords(mc1gt_1,94,83,95).  visible(mc1gt_1,95).
orientation(mc1gt_0,148,96). coords(mc1gt_0,79,96,96).  visible(mc1gt_0,96). orientation(mc1gt_1,145,96). coords(mc1gt_1,94,83,96).  visible(mc1gt_1,96).
orientation(mc1gt_0,148,97). coords(mc1gt_0,79,96,97).  visible(mc1gt_0,97). orientation(mc1gt_1,145,97). coords(mc1gt_1,94,83,97).  visible(mc1gt_1,97).
orientation(mc1gt_0,148,98). coords(mc1gt_0,79,96,98).  visible(mc1gt_0,98). orientation(mc1gt_1,145,98). coords(mc1gt_1,94,83,98).  visible(mc1gt_1,98).
orientation(mc1gt_0,148,99). coords(mc1gt_0,79,96,99).  visible(mc1gt_0,99). orientation(mc1gt_1,145,99). coords(mc1gt_1,94,83,99).  visible(mc1gt_1,99).
orientation(mc1gt_0,148,100). coords(mc1gt_0,79,96,100).  visible(mc1gt_0,100). orientation(mc1gt_1,145,100). coords(mc1gt_1,94,83,100).  visible(mc1gt_1,100).
orientation(mc1gt_0,148,101). coords(mc1gt_0,79,96,101).  visible(mc1gt_0,101). orientation(mc1gt_1,145,101). coords(mc1gt_1,94,83,101).  visible(mc1gt_1,101).
happensAt(disappear(mc1gt_0),100). happensAt(disappear(mc1gt_1),100).
1{simpleEvent(T,mc1gt_0,active); simpleEvent(T,mc1gt_0,inactive); simpleEvent(T,mc1gt_0,walking); simpleEvent(T,mc1gt_0,running)}1 :- T=76..100.
1{simpleEvent(T,mc1gt_1,active); simpleEvent(T,mc1gt_1,inactive); simpleEvent(T,mc1gt_1,walking); simpleEvent(T,mc1gt_1,running)}1 :- T=76..100.
:- holdsAt(moving(mc1gt_0,mc1gt_1),76). :- holdsAt(moving(mc1gt_1,mc1gt_0),76).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),77). :- not holdsAt(moving(mc1gt_1,mc1gt_0),77).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),78). :- not holdsAt(moving(mc1gt_1,mc1gt_0),78).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),79). :- not holdsAt(moving(mc1gt_1,mc1gt_0),79).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),80). :- not holdsAt(moving(mc1gt_1,mc1gt_0),80).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),81). :- not holdsAt(moving(mc1gt_1,mc1gt_0),81).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),82). :- not holdsAt(moving(mc1gt_1,mc1gt_0),82).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),83). :- not holdsAt(moving(mc1gt_1,mc1gt_0),83).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),84). :- not holdsAt(moving(mc1gt_1,mc1gt_0),84).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),85). :- not holdsAt(moving(mc1gt_1,mc1gt_0),85).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),86). :- not holdsAt(moving(mc1gt_1,mc1gt_0),86).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),87). :- not holdsAt(moving(mc1gt_1,mc1gt_0),87).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),88). :- not holdsAt(moving(mc1gt_1,mc1gt_0),88).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),89). :- not holdsAt(moving(mc1gt_1,mc1gt_0),89).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),90). :- not holdsAt(moving(mc1gt_1,mc1gt_0),90).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),91). :- not holdsAt(moving(mc1gt_1,mc1gt_0),91).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),92). :- not holdsAt(moving(mc1gt_1,mc1gt_0),92).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),93). :- not holdsAt(moving(mc1gt_1,mc1gt_0),93).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),94). :- not holdsAt(moving(mc1gt_1,mc1gt_0),94).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),95). :- not holdsAt(moving(mc1gt_1,mc1gt_0),95).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),96). :- not holdsAt(moving(mc1gt_1,mc1gt_0),96).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),97). :- not holdsAt(moving(mc1gt_1,mc1gt_0),97).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),98). :- not holdsAt(moving(mc1gt_1,mc1gt_0),98).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),99). :- not holdsAt(moving(mc1gt_1,mc1gt_0),99).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),100). :- not holdsAt(moving(mc1gt_1,mc1gt_0),100).
:- holdsAt(moving(mc1gt_0,mc1gt_1),101). :- holdsAt(moving(mc1gt_1,mc1gt_0),101).
:- simpleEvent(T,P,inactive), T>76 , T<100.
:- simpleEvent(T,P,running), T>76 , T<100.

#show simpleEvent/3.