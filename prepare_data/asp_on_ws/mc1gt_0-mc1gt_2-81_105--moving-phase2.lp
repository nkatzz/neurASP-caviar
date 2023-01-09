

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

	person(mc1gt_0;mc1gt_2).
orientation(mc1gt_0,148,81). coords(mc1gt_0,79,96,81).  visible(mc1gt_0,81). orientation(mc1gt_2,144,81). coords(mc1gt_2,78,65,81).  visible(mc1gt_2,81).
orientation(mc1gt_0,148,82). coords(mc1gt_0,79,96,82).  visible(mc1gt_0,82). orientation(mc1gt_2,144,82). coords(mc1gt_2,78,65,82).  visible(mc1gt_2,82).
orientation(mc1gt_0,148,83). coords(mc1gt_0,79,96,83).  visible(mc1gt_0,83). orientation(mc1gt_2,144,83). coords(mc1gt_2,78,65,83).  visible(mc1gt_2,83).
orientation(mc1gt_0,148,84). coords(mc1gt_0,79,96,84).  visible(mc1gt_0,84). orientation(mc1gt_2,144,84). coords(mc1gt_2,78,65,84).  visible(mc1gt_2,84).
orientation(mc1gt_0,148,85). coords(mc1gt_0,79,96,85).  visible(mc1gt_0,85). orientation(mc1gt_2,144,85). coords(mc1gt_2,78,65,85).  visible(mc1gt_2,85).
orientation(mc1gt_0,148,86). coords(mc1gt_0,79,96,86).  visible(mc1gt_0,86). orientation(mc1gt_2,144,86). coords(mc1gt_2,78,65,86).  visible(mc1gt_2,86).
orientation(mc1gt_0,148,87). coords(mc1gt_0,79,96,87).  visible(mc1gt_0,87). orientation(mc1gt_2,144,87). coords(mc1gt_2,78,65,87).  visible(mc1gt_2,87).
orientation(mc1gt_0,148,88). coords(mc1gt_0,79,96,88).  visible(mc1gt_0,88). orientation(mc1gt_2,144,88). coords(mc1gt_2,78,65,88).  visible(mc1gt_2,88).
orientation(mc1gt_0,148,89). coords(mc1gt_0,79,96,89).  visible(mc1gt_0,89). orientation(mc1gt_2,144,89). coords(mc1gt_2,78,65,89).  visible(mc1gt_2,89).
orientation(mc1gt_0,148,90). coords(mc1gt_0,79,96,90).  visible(mc1gt_0,90). orientation(mc1gt_2,144,90). coords(mc1gt_2,78,65,90).  visible(mc1gt_2,90).
orientation(mc1gt_0,148,91). coords(mc1gt_0,79,96,91).  visible(mc1gt_0,91). orientation(mc1gt_2,144,91). coords(mc1gt_2,78,65,91).  visible(mc1gt_2,91).
orientation(mc1gt_0,148,92). coords(mc1gt_0,79,96,92).  visible(mc1gt_0,92). orientation(mc1gt_2,144,92). coords(mc1gt_2,78,65,92).  visible(mc1gt_2,92).
orientation(mc1gt_0,148,93). coords(mc1gt_0,79,96,93).  visible(mc1gt_0,93). orientation(mc1gt_2,144,93). coords(mc1gt_2,78,65,93).  visible(mc1gt_2,93).
orientation(mc1gt_0,148,94). coords(mc1gt_0,79,96,94).  visible(mc1gt_0,94). orientation(mc1gt_2,144,94). coords(mc1gt_2,78,65,94).  visible(mc1gt_2,94).
orientation(mc1gt_0,148,95). coords(mc1gt_0,79,96,95).  visible(mc1gt_0,95). orientation(mc1gt_2,144,95). coords(mc1gt_2,78,65,95).  visible(mc1gt_2,95).
orientation(mc1gt_0,148,96). coords(mc1gt_0,79,96,96).  visible(mc1gt_0,96). orientation(mc1gt_2,144,96). coords(mc1gt_2,78,65,96).  visible(mc1gt_2,96).
orientation(mc1gt_0,148,97). coords(mc1gt_0,79,96,97).  visible(mc1gt_0,97). orientation(mc1gt_2,144,97). coords(mc1gt_2,78,65,97).  visible(mc1gt_2,97).
orientation(mc1gt_0,148,98). coords(mc1gt_0,79,96,98).  visible(mc1gt_0,98). orientation(mc1gt_2,144,98). coords(mc1gt_2,78,65,98).  visible(mc1gt_2,98).
orientation(mc1gt_0,148,99). coords(mc1gt_0,79,96,99).  visible(mc1gt_0,99). orientation(mc1gt_2,144,99). coords(mc1gt_2,78,65,99).  visible(mc1gt_2,99).
orientation(mc1gt_0,148,100). coords(mc1gt_0,79,96,100).  visible(mc1gt_0,100). orientation(mc1gt_2,144,100). coords(mc1gt_2,78,65,100).  visible(mc1gt_2,100).
orientation(mc1gt_0,148,101). coords(mc1gt_0,79,97,101).  visible(mc1gt_0,101). orientation(mc1gt_2,144,101). coords(mc1gt_2,77,66,101).  visible(mc1gt_2,101).
orientation(mc1gt_0,148,102). coords(mc1gt_0,79,97,102).  visible(mc1gt_0,102). orientation(mc1gt_2,144,102). coords(mc1gt_2,77,66,102).  visible(mc1gt_2,102).
orientation(mc1gt_0,148,103). coords(mc1gt_0,79,97,103).  visible(mc1gt_0,103). orientation(mc1gt_2,144,103). coords(mc1gt_2,77,67,103).  visible(mc1gt_2,103).
orientation(mc1gt_0,148,104). coords(mc1gt_0,80,97,104).  visible(mc1gt_0,104). orientation(mc1gt_2,144,104). coords(mc1gt_2,76,67,104).  visible(mc1gt_2,104).
orientation(mc1gt_0,148,105). coords(mc1gt_0,80,97,105).  visible(mc1gt_0,105). orientation(mc1gt_2,144,105). coords(mc1gt_2,76,68,105).  visible(mc1gt_2,105).
orientation(mc1gt_0,148,106). coords(mc1gt_0,80,97,106).  visible(mc1gt_0,106). orientation(mc1gt_2,144,106). coords(mc1gt_2,76,68,106).  visible(mc1gt_2,106).
happensAt(disappear(mc1gt_0),105). happensAt(disappear(mc1gt_2),105).
1{simpleEvent(T,mc1gt_0,active); simpleEvent(T,mc1gt_0,inactive); simpleEvent(T,mc1gt_0,walking); simpleEvent(T,mc1gt_0,running)}1 :- T=81..105.
1{simpleEvent(T,mc1gt_2,active); simpleEvent(T,mc1gt_2,inactive); simpleEvent(T,mc1gt_2,walking); simpleEvent(T,mc1gt_2,running)}1 :- T=81..105.
:- holdsAt(moving(mc1gt_0,mc1gt_2),81). :- holdsAt(moving(mc1gt_2,mc1gt_0),81).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),82). :- not holdsAt(moving(mc1gt_2,mc1gt_0),82).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),83). :- not holdsAt(moving(mc1gt_2,mc1gt_0),83).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),84). :- not holdsAt(moving(mc1gt_2,mc1gt_0),84).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),85). :- not holdsAt(moving(mc1gt_2,mc1gt_0),85).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),86). :- not holdsAt(moving(mc1gt_2,mc1gt_0),86).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),87). :- not holdsAt(moving(mc1gt_2,mc1gt_0),87).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),88). :- not holdsAt(moving(mc1gt_2,mc1gt_0),88).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),89). :- not holdsAt(moving(mc1gt_2,mc1gt_0),89).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),90). :- not holdsAt(moving(mc1gt_2,mc1gt_0),90).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),91). :- not holdsAt(moving(mc1gt_2,mc1gt_0),91).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),92). :- not holdsAt(moving(mc1gt_2,mc1gt_0),92).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),93). :- not holdsAt(moving(mc1gt_2,mc1gt_0),93).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),94). :- not holdsAt(moving(mc1gt_2,mc1gt_0),94).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),95). :- not holdsAt(moving(mc1gt_2,mc1gt_0),95).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),96). :- not holdsAt(moving(mc1gt_2,mc1gt_0),96).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),97). :- not holdsAt(moving(mc1gt_2,mc1gt_0),97).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),98). :- not holdsAt(moving(mc1gt_2,mc1gt_0),98).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),99). :- not holdsAt(moving(mc1gt_2,mc1gt_0),99).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),100). :- not holdsAt(moving(mc1gt_2,mc1gt_0),100).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),101). :- not holdsAt(moving(mc1gt_2,mc1gt_0),101).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),102). :- not holdsAt(moving(mc1gt_2,mc1gt_0),102).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),103). :- not holdsAt(moving(mc1gt_2,mc1gt_0),103).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),104). :- not holdsAt(moving(mc1gt_2,mc1gt_0),104).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),105). :- not holdsAt(moving(mc1gt_2,mc1gt_0),105).
:- holdsAt(moving(mc1gt_0,mc1gt_2),106). :- holdsAt(moving(mc1gt_2,mc1gt_0),106).
:- simpleEvent(T,P,inactive), T>81 , T<105.
:- simpleEvent(T,P,running), T>81 , T<105.

#show simpleEvent/3.