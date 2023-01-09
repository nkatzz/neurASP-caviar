

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

	person(wk1gt_4;wk1gt_5).
orientation(wk1gt_4,0,75). coords(wk1gt_4,36,198,75).  visible(wk1gt_4,75). orientation(wk1gt_5,0,75). coords(wk1gt_5,27,204,75).  visible(wk1gt_5,75).
orientation(wk1gt_4,0,76). coords(wk1gt_4,36,198,76).  visible(wk1gt_4,76). orientation(wk1gt_5,0,76). coords(wk1gt_5,28,204,76).  visible(wk1gt_5,76).
orientation(wk1gt_4,0,77). coords(wk1gt_4,36,197,77).  visible(wk1gt_4,77). orientation(wk1gt_5,0,77). coords(wk1gt_5,28,203,77).  visible(wk1gt_5,77).
orientation(wk1gt_4,0,78). coords(wk1gt_4,37,197,78).  visible(wk1gt_4,78). orientation(wk1gt_5,0,78). coords(wk1gt_5,28,203,78).  visible(wk1gt_5,78).
orientation(wk1gt_4,0,79). coords(wk1gt_4,38,197,79).  visible(wk1gt_4,79). orientation(wk1gt_5,0,79). coords(wk1gt_5,28,202,79).  visible(wk1gt_5,79).
orientation(wk1gt_4,0,80). coords(wk1gt_4,38,196,80).  visible(wk1gt_4,80). orientation(wk1gt_5,0,80). coords(wk1gt_5,29,201,80).  visible(wk1gt_5,80).
orientation(wk1gt_4,0,81). coords(wk1gt_4,38,196,81).  visible(wk1gt_4,81). orientation(wk1gt_5,0,81). coords(wk1gt_5,29,201,81).  visible(wk1gt_5,81).
orientation(wk1gt_4,0,82). coords(wk1gt_4,38,196,82).  visible(wk1gt_4,82). orientation(wk1gt_5,0,82). coords(wk1gt_5,29,200,82).  visible(wk1gt_5,82).
orientation(wk1gt_4,0,83). coords(wk1gt_4,39,196,83).  visible(wk1gt_4,83). orientation(wk1gt_5,0,83). coords(wk1gt_5,29,198,83).  visible(wk1gt_5,83).
orientation(wk1gt_4,0,84). coords(wk1gt_4,39,196,84).  visible(wk1gt_4,84). orientation(wk1gt_5,0,84). coords(wk1gt_5,30,197,84).  visible(wk1gt_5,84).
orientation(wk1gt_4,0,85). coords(wk1gt_4,39,196,85).  visible(wk1gt_4,85). orientation(wk1gt_5,0,85). coords(wk1gt_5,30,197,85).  visible(wk1gt_5,85).
orientation(wk1gt_4,0,86). coords(wk1gt_4,39,195,86).  visible(wk1gt_4,86). orientation(wk1gt_5,0,86). coords(wk1gt_5,31,197,86).  visible(wk1gt_5,86).
orientation(wk1gt_4,0,87). coords(wk1gt_4,39,195,87).  visible(wk1gt_4,87). orientation(wk1gt_5,0,87). coords(wk1gt_5,31,197,87).  visible(wk1gt_5,87).
orientation(wk1gt_4,0,88). coords(wk1gt_4,39,194,88).  visible(wk1gt_4,88). orientation(wk1gt_5,0,88). coords(wk1gt_5,32,197,88).  visible(wk1gt_5,88).
orientation(wk1gt_4,0,89). coords(wk1gt_4,39,194,89).  visible(wk1gt_4,89). orientation(wk1gt_5,0,89). coords(wk1gt_5,32,196,89).  visible(wk1gt_5,89).
orientation(wk1gt_4,0,90). coords(wk1gt_4,39,194,90).  visible(wk1gt_4,90). orientation(wk1gt_5,0,90). coords(wk1gt_5,32,195,90).  visible(wk1gt_5,90).
orientation(wk1gt_4,0,91). coords(wk1gt_4,39,194,91).  visible(wk1gt_4,91). orientation(wk1gt_5,0,91). coords(wk1gt_5,32,195,91).  visible(wk1gt_5,91).
orientation(wk1gt_4,0,92). coords(wk1gt_4,39,194,92).  visible(wk1gt_4,92). orientation(wk1gt_5,0,92). coords(wk1gt_5,32,194,92).  visible(wk1gt_5,92).
orientation(wk1gt_4,0,93). coords(wk1gt_4,39,194,93).  visible(wk1gt_4,93). orientation(wk1gt_5,0,93). coords(wk1gt_5,32,194,93).  visible(wk1gt_5,93).
orientation(wk1gt_4,0,94). coords(wk1gt_4,39,194,94).  visible(wk1gt_4,94). orientation(wk1gt_5,0,94). coords(wk1gt_5,32,193,94).  visible(wk1gt_5,94).
orientation(wk1gt_4,0,95). coords(wk1gt_4,39,194,95).  visible(wk1gt_4,95). orientation(wk1gt_5,0,95). coords(wk1gt_5,32,192,95).  visible(wk1gt_5,95).
orientation(wk1gt_4,0,96). coords(wk1gt_4,39,194,96).  visible(wk1gt_4,96). orientation(wk1gt_5,0,96). coords(wk1gt_5,32,191,96).  visible(wk1gt_5,96).
orientation(wk1gt_4,0,97). coords(wk1gt_4,41,194,97).  visible(wk1gt_4,97). orientation(wk1gt_5,0,97). coords(wk1gt_5,34,191,97).  visible(wk1gt_5,97).
orientation(wk1gt_4,0,98). coords(wk1gt_4,41,194,98).  visible(wk1gt_4,98). orientation(wk1gt_5,0,98). coords(wk1gt_5,35,189,98).  visible(wk1gt_5,98).
orientation(wk1gt_4,0,99). coords(wk1gt_4,41,193,99).  visible(wk1gt_4,99). orientation(wk1gt_5,0,99). coords(wk1gt_5,35,188,99).  visible(wk1gt_5,99).
orientation(wk1gt_4,0,100). coords(wk1gt_4,41,193,100).  visible(wk1gt_4,100). orientation(wk1gt_5,0,100). coords(wk1gt_5,35,188,100).  visible(wk1gt_5,100).
happensAt(disappear(wk1gt_4),99). happensAt(disappear(wk1gt_5),99).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=75..99.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=75..99.
:- holdsAt(moving(wk1gt_4,wk1gt_5),75). :- holdsAt(moving(wk1gt_5,wk1gt_4),75).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),76). :- not holdsAt(moving(wk1gt_5,wk1gt_4),76).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),77). :- not holdsAt(moving(wk1gt_5,wk1gt_4),77).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),78). :- not holdsAt(moving(wk1gt_5,wk1gt_4),78).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),79). :- not holdsAt(moving(wk1gt_5,wk1gt_4),79).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),80). :- not holdsAt(moving(wk1gt_5,wk1gt_4),80).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),81). :- not holdsAt(moving(wk1gt_5,wk1gt_4),81).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),82). :- not holdsAt(moving(wk1gt_5,wk1gt_4),82).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),83). :- not holdsAt(moving(wk1gt_5,wk1gt_4),83).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),84). :- not holdsAt(moving(wk1gt_5,wk1gt_4),84).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),85). :- not holdsAt(moving(wk1gt_5,wk1gt_4),85).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),86). :- not holdsAt(moving(wk1gt_5,wk1gt_4),86).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),87). :- not holdsAt(moving(wk1gt_5,wk1gt_4),87).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),88). :- not holdsAt(moving(wk1gt_5,wk1gt_4),88).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),89). :- not holdsAt(moving(wk1gt_5,wk1gt_4),89).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),90). :- not holdsAt(moving(wk1gt_5,wk1gt_4),90).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),91). :- not holdsAt(moving(wk1gt_5,wk1gt_4),91).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),92). :- not holdsAt(moving(wk1gt_5,wk1gt_4),92).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),93). :- not holdsAt(moving(wk1gt_5,wk1gt_4),93).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),94). :- not holdsAt(moving(wk1gt_5,wk1gt_4),94).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),95). :- not holdsAt(moving(wk1gt_5,wk1gt_4),95).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),96). :- not holdsAt(moving(wk1gt_5,wk1gt_4),96).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),97). :- not holdsAt(moving(wk1gt_5,wk1gt_4),97).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),98). :- not holdsAt(moving(wk1gt_5,wk1gt_4),98).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),99). :- not holdsAt(moving(wk1gt_5,wk1gt_4),99).
:- holdsAt(moving(wk1gt_4,wk1gt_5),100). :- holdsAt(moving(wk1gt_5,wk1gt_4),100).
:- simpleEvent(T,P,inactive), T>75 , T<99.
:- simpleEvent(T,P,running), T>75 , T<99.

#show simpleEvent/3.