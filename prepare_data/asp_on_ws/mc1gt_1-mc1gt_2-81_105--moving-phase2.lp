

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

	person(mc1gt_1;mc1gt_2).
orientation(mc1gt_1,145,81). coords(mc1gt_1,94,83,81).  visible(mc1gt_1,81). orientation(mc1gt_2,144,81). coords(mc1gt_2,78,65,81).  visible(mc1gt_2,81).
orientation(mc1gt_1,145,82). coords(mc1gt_1,94,83,82).  visible(mc1gt_1,82). orientation(mc1gt_2,144,82). coords(mc1gt_2,78,65,82).  visible(mc1gt_2,82).
orientation(mc1gt_1,145,83). coords(mc1gt_1,94,83,83).  visible(mc1gt_1,83). orientation(mc1gt_2,144,83). coords(mc1gt_2,78,65,83).  visible(mc1gt_2,83).
orientation(mc1gt_1,145,84). coords(mc1gt_1,94,83,84).  visible(mc1gt_1,84). orientation(mc1gt_2,144,84). coords(mc1gt_2,78,65,84).  visible(mc1gt_2,84).
orientation(mc1gt_1,145,85). coords(mc1gt_1,94,83,85).  visible(mc1gt_1,85). orientation(mc1gt_2,144,85). coords(mc1gt_2,78,65,85).  visible(mc1gt_2,85).
orientation(mc1gt_1,145,86). coords(mc1gt_1,94,83,86).  visible(mc1gt_1,86). orientation(mc1gt_2,144,86). coords(mc1gt_2,78,65,86).  visible(mc1gt_2,86).
orientation(mc1gt_1,145,87). coords(mc1gt_1,94,83,87).  visible(mc1gt_1,87). orientation(mc1gt_2,144,87). coords(mc1gt_2,78,65,87).  visible(mc1gt_2,87).
orientation(mc1gt_1,145,88). coords(mc1gt_1,94,83,88).  visible(mc1gt_1,88). orientation(mc1gt_2,144,88). coords(mc1gt_2,78,65,88).  visible(mc1gt_2,88).
orientation(mc1gt_1,145,89). coords(mc1gt_1,94,83,89).  visible(mc1gt_1,89). orientation(mc1gt_2,144,89). coords(mc1gt_2,78,65,89).  visible(mc1gt_2,89).
orientation(mc1gt_1,145,90). coords(mc1gt_1,94,83,90).  visible(mc1gt_1,90). orientation(mc1gt_2,144,90). coords(mc1gt_2,78,65,90).  visible(mc1gt_2,90).
orientation(mc1gt_1,145,91). coords(mc1gt_1,94,83,91).  visible(mc1gt_1,91). orientation(mc1gt_2,144,91). coords(mc1gt_2,78,65,91).  visible(mc1gt_2,91).
orientation(mc1gt_1,145,92). coords(mc1gt_1,94,83,92).  visible(mc1gt_1,92). orientation(mc1gt_2,144,92). coords(mc1gt_2,78,65,92).  visible(mc1gt_2,92).
orientation(mc1gt_1,145,93). coords(mc1gt_1,94,83,93).  visible(mc1gt_1,93). orientation(mc1gt_2,144,93). coords(mc1gt_2,78,65,93).  visible(mc1gt_2,93).
orientation(mc1gt_1,145,94). coords(mc1gt_1,94,83,94).  visible(mc1gt_1,94). orientation(mc1gt_2,144,94). coords(mc1gt_2,78,65,94).  visible(mc1gt_2,94).
orientation(mc1gt_1,145,95). coords(mc1gt_1,94,83,95).  visible(mc1gt_1,95). orientation(mc1gt_2,144,95). coords(mc1gt_2,78,65,95).  visible(mc1gt_2,95).
orientation(mc1gt_1,145,96). coords(mc1gt_1,94,83,96).  visible(mc1gt_1,96). orientation(mc1gt_2,144,96). coords(mc1gt_2,78,65,96).  visible(mc1gt_2,96).
orientation(mc1gt_1,145,97). coords(mc1gt_1,94,83,97).  visible(mc1gt_1,97). orientation(mc1gt_2,144,97). coords(mc1gt_2,78,65,97).  visible(mc1gt_2,97).
orientation(mc1gt_1,145,98). coords(mc1gt_1,94,83,98).  visible(mc1gt_1,98). orientation(mc1gt_2,144,98). coords(mc1gt_2,78,65,98).  visible(mc1gt_2,98).
orientation(mc1gt_1,145,99). coords(mc1gt_1,94,83,99).  visible(mc1gt_1,99). orientation(mc1gt_2,144,99). coords(mc1gt_2,78,65,99).  visible(mc1gt_2,99).
orientation(mc1gt_1,145,100). coords(mc1gt_1,94,83,100).  visible(mc1gt_1,100). orientation(mc1gt_2,144,100). coords(mc1gt_2,78,65,100).  visible(mc1gt_2,100).
orientation(mc1gt_1,145,101). coords(mc1gt_1,94,83,101).  visible(mc1gt_1,101). orientation(mc1gt_2,144,101). coords(mc1gt_2,77,66,101).  visible(mc1gt_2,101).
orientation(mc1gt_1,145,102). coords(mc1gt_1,95,82,102).  visible(mc1gt_1,102). orientation(mc1gt_2,144,102). coords(mc1gt_2,77,66,102).  visible(mc1gt_2,102).
orientation(mc1gt_1,145,103). coords(mc1gt_1,95,82,103).  visible(mc1gt_1,103). orientation(mc1gt_2,144,103). coords(mc1gt_2,77,67,103).  visible(mc1gt_2,103).
orientation(mc1gt_1,145,104). coords(mc1gt_1,95,82,104).  visible(mc1gt_1,104). orientation(mc1gt_2,144,104). coords(mc1gt_2,76,67,104).  visible(mc1gt_2,104).
orientation(mc1gt_1,145,105). coords(mc1gt_1,95,82,105).  visible(mc1gt_1,105). orientation(mc1gt_2,144,105). coords(mc1gt_2,76,68,105).  visible(mc1gt_2,105).
orientation(mc1gt_1,145,106). coords(mc1gt_1,95,82,106).  visible(mc1gt_1,106). orientation(mc1gt_2,144,106). coords(mc1gt_2,76,68,106).  visible(mc1gt_2,106).
happensAt(disappear(mc1gt_1),105). happensAt(disappear(mc1gt_2),105).
1{simpleEvent(T,mc1gt_1,active); simpleEvent(T,mc1gt_1,inactive); simpleEvent(T,mc1gt_1,walking); simpleEvent(T,mc1gt_1,running)}1 :- T=81..105.
1{simpleEvent(T,mc1gt_2,active); simpleEvent(T,mc1gt_2,inactive); simpleEvent(T,mc1gt_2,walking); simpleEvent(T,mc1gt_2,running)}1 :- T=81..105.
:- holdsAt(moving(mc1gt_1,mc1gt_2),81). :- holdsAt(moving(mc1gt_2,mc1gt_1),81).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),82). :- not holdsAt(moving(mc1gt_2,mc1gt_1),82).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),83). :- not holdsAt(moving(mc1gt_2,mc1gt_1),83).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),84). :- not holdsAt(moving(mc1gt_2,mc1gt_1),84).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),85). :- not holdsAt(moving(mc1gt_2,mc1gt_1),85).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),86). :- not holdsAt(moving(mc1gt_2,mc1gt_1),86).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),87). :- not holdsAt(moving(mc1gt_2,mc1gt_1),87).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),88). :- not holdsAt(moving(mc1gt_2,mc1gt_1),88).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),89). :- not holdsAt(moving(mc1gt_2,mc1gt_1),89).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),90). :- not holdsAt(moving(mc1gt_2,mc1gt_1),90).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),91). :- not holdsAt(moving(mc1gt_2,mc1gt_1),91).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),92). :- not holdsAt(moving(mc1gt_2,mc1gt_1),92).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),93). :- not holdsAt(moving(mc1gt_2,mc1gt_1),93).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),94). :- not holdsAt(moving(mc1gt_2,mc1gt_1),94).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),95). :- not holdsAt(moving(mc1gt_2,mc1gt_1),95).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),96). :- not holdsAt(moving(mc1gt_2,mc1gt_1),96).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),97). :- not holdsAt(moving(mc1gt_2,mc1gt_1),97).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),98). :- not holdsAt(moving(mc1gt_2,mc1gt_1),98).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),99). :- not holdsAt(moving(mc1gt_2,mc1gt_1),99).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),100). :- not holdsAt(moving(mc1gt_2,mc1gt_1),100).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),101). :- not holdsAt(moving(mc1gt_2,mc1gt_1),101).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),102). :- not holdsAt(moving(mc1gt_2,mc1gt_1),102).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),103). :- not holdsAt(moving(mc1gt_2,mc1gt_1),103).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),104). :- not holdsAt(moving(mc1gt_2,mc1gt_1),104).
:- not holdsAt(moving(mc1gt_1,mc1gt_2),105). :- not holdsAt(moving(mc1gt_2,mc1gt_1),105).
:- holdsAt(moving(mc1gt_1,mc1gt_2),106). :- holdsAt(moving(mc1gt_2,mc1gt_1),106).
:- simpleEvent(T,P,inactive), T>81 , T<105.
:- simpleEvent(T,P,running), T>81 , T<105.

#show simpleEvent/3.