

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
orientation(spgt_0,152,84). coords(spgt_0,64,155,84).  visible(spgt_0,84). orientation(spgt_1,160,84). coords(spgt_1,64,145,84).  visible(spgt_1,84).
orientation(spgt_0,152,85). coords(spgt_0,64,155,85).  visible(spgt_0,85). orientation(spgt_1,160,85). coords(spgt_1,65,144,85).  visible(spgt_1,85).
orientation(spgt_0,152,86). coords(spgt_0,64,154,86).  visible(spgt_0,86). orientation(spgt_1,160,86). coords(spgt_1,65,144,86).  visible(spgt_1,86).
orientation(spgt_0,152,87). coords(spgt_0,65,154,87).  visible(spgt_0,87). orientation(spgt_1,160,87). coords(spgt_1,65,143,87).  visible(spgt_1,87).
orientation(spgt_0,152,88). coords(spgt_0,66,152,88).  visible(spgt_0,88). orientation(spgt_1,160,88). coords(spgt_1,66,143,88).  visible(spgt_1,88).
orientation(spgt_0,152,89). coords(spgt_0,67,152,89).  visible(spgt_0,89). orientation(spgt_1,160,89). coords(spgt_1,67,142,89).  visible(spgt_1,89).
orientation(spgt_0,152,90). coords(spgt_0,68,152,90).  visible(spgt_0,90). orientation(spgt_1,160,90). coords(spgt_1,68,142,90).  visible(spgt_1,90).
orientation(spgt_0,152,91). coords(spgt_0,68,152,91).  visible(spgt_0,91). orientation(spgt_1,160,91). coords(spgt_1,69,142,91).  visible(spgt_1,91).
orientation(spgt_0,152,92). coords(spgt_0,68,152,92).  visible(spgt_0,92). orientation(spgt_1,160,92). coords(spgt_1,69,142,92).  visible(spgt_1,92).
orientation(spgt_0,152,93). coords(spgt_0,70,151,93).  visible(spgt_0,93). orientation(spgt_1,154,93). coords(spgt_1,70,141,93).  visible(spgt_1,93).
orientation(spgt_0,152,94). coords(spgt_0,70,150,94).  visible(spgt_0,94). orientation(spgt_1,154,94). coords(spgt_1,71,140,94).  visible(spgt_1,94).
orientation(spgt_0,152,95). coords(spgt_0,70,150,95).  visible(spgt_0,95). orientation(spgt_1,154,95). coords(spgt_1,71,140,95).  visible(spgt_1,95).
orientation(spgt_0,152,96). coords(spgt_0,70,150,96).  visible(spgt_0,96). orientation(spgt_1,154,96). coords(spgt_1,72,140,96).  visible(spgt_1,96).
orientation(spgt_0,152,97). coords(spgt_0,70,149,97).  visible(spgt_0,97). orientation(spgt_1,154,97). coords(spgt_1,72,139,97).  visible(spgt_1,97).
orientation(spgt_0,152,98). coords(spgt_0,70,149,98).  visible(spgt_0,98). orientation(spgt_1,154,98). coords(spgt_1,72,138,98).  visible(spgt_1,98).
orientation(spgt_0,152,99). coords(spgt_0,70,148,99).  visible(spgt_0,99). orientation(spgt_1,154,99). coords(spgt_1,72,138,99).  visible(spgt_1,99).
orientation(spgt_0,152,100). coords(spgt_0,70,148,100).  visible(spgt_0,100). orientation(spgt_1,154,100). coords(spgt_1,72,138,100).  visible(spgt_1,100).
orientation(spgt_0,152,101). coords(spgt_0,73,148,101).  visible(spgt_0,101). orientation(spgt_1,154,101). coords(spgt_1,72,137,101).  visible(spgt_1,101).
orientation(spgt_0,152,102). coords(spgt_0,74,148,102).  visible(spgt_0,102). orientation(spgt_1,154,102). coords(spgt_1,73,137,102).  visible(spgt_1,102).
orientation(spgt_0,152,103). coords(spgt_0,75,148,103).  visible(spgt_0,103). orientation(spgt_1,154,103). coords(spgt_1,75,137,103).  visible(spgt_1,103).
orientation(spgt_0,152,104). coords(spgt_0,76,147,104).  visible(spgt_0,104). orientation(spgt_1,154,104). coords(spgt_1,76,136,104).  visible(spgt_1,104).
orientation(spgt_0,152,105). coords(spgt_0,77,147,105).  visible(spgt_0,105). orientation(spgt_1,154,105). coords(spgt_1,77,136,105).  visible(spgt_1,105).
orientation(spgt_0,152,106). coords(spgt_0,77,147,106).  visible(spgt_0,106). orientation(spgt_1,154,106). coords(spgt_1,78,135,106).  visible(spgt_1,106).
orientation(spgt_0,152,107). coords(spgt_0,77,147,107).  visible(spgt_0,107). orientation(spgt_1,154,107). coords(spgt_1,78,134,107).  visible(spgt_1,107).
orientation(spgt_0,152,108). coords(spgt_0,77,147,108).  visible(spgt_0,108). orientation(spgt_1,154,108). coords(spgt_1,78,134,108).  visible(spgt_1,108).
orientation(spgt_0,152,109). coords(spgt_0,77,147,109).  visible(spgt_0,109). orientation(spgt_1,154,109). coords(spgt_1,78,134,109).  visible(spgt_1,109).
happensAt(disappear(spgt_0),108). happensAt(disappear(spgt_1),108).
1{simpleEvent(T,spgt_0,active); simpleEvent(T,spgt_0,inactive); simpleEvent(T,spgt_0,walking); simpleEvent(T,spgt_0,running)}1 :- T=84..108.
1{simpleEvent(T,spgt_1,active); simpleEvent(T,spgt_1,inactive); simpleEvent(T,spgt_1,walking); simpleEvent(T,spgt_1,running)}1 :- T=84..108.
:- holdsAt(moving(spgt_0,spgt_1),84). :- holdsAt(moving(spgt_1,spgt_0),84).
:- not holdsAt(moving(spgt_0,spgt_1),85). :- not holdsAt(moving(spgt_1,spgt_0),85).
:- not holdsAt(moving(spgt_0,spgt_1),86). :- not holdsAt(moving(spgt_1,spgt_0),86).
:- not holdsAt(moving(spgt_0,spgt_1),87). :- not holdsAt(moving(spgt_1,spgt_0),87).
:- not holdsAt(moving(spgt_0,spgt_1),88). :- not holdsAt(moving(spgt_1,spgt_0),88).
:- not holdsAt(moving(spgt_0,spgt_1),89). :- not holdsAt(moving(spgt_1,spgt_0),89).
:- not holdsAt(moving(spgt_0,spgt_1),90). :- not holdsAt(moving(spgt_1,spgt_0),90).
:- not holdsAt(moving(spgt_0,spgt_1),91). :- not holdsAt(moving(spgt_1,spgt_0),91).
:- not holdsAt(moving(spgt_0,spgt_1),92). :- not holdsAt(moving(spgt_1,spgt_0),92).
:- not holdsAt(moving(spgt_0,spgt_1),93). :- not holdsAt(moving(spgt_1,spgt_0),93).
:- not holdsAt(moving(spgt_0,spgt_1),94). :- not holdsAt(moving(spgt_1,spgt_0),94).
:- not holdsAt(moving(spgt_0,spgt_1),95). :- not holdsAt(moving(spgt_1,spgt_0),95).
:- not holdsAt(moving(spgt_0,spgt_1),96). :- not holdsAt(moving(spgt_1,spgt_0),96).
:- not holdsAt(moving(spgt_0,spgt_1),97). :- not holdsAt(moving(spgt_1,spgt_0),97).
:- not holdsAt(moving(spgt_0,spgt_1),98). :- not holdsAt(moving(spgt_1,spgt_0),98).
:- not holdsAt(moving(spgt_0,spgt_1),99). :- not holdsAt(moving(spgt_1,spgt_0),99).
:- not holdsAt(moving(spgt_0,spgt_1),100). :- not holdsAt(moving(spgt_1,spgt_0),100).
:- not holdsAt(moving(spgt_0,spgt_1),101). :- not holdsAt(moving(spgt_1,spgt_0),101).
:- not holdsAt(moving(spgt_0,spgt_1),102). :- not holdsAt(moving(spgt_1,spgt_0),102).
:- not holdsAt(moving(spgt_0,spgt_1),103). :- not holdsAt(moving(spgt_1,spgt_0),103).
:- not holdsAt(moving(spgt_0,spgt_1),104). :- not holdsAt(moving(spgt_1,spgt_0),104).
:- not holdsAt(moving(spgt_0,spgt_1),105). :- not holdsAt(moving(spgt_1,spgt_0),105).
:- not holdsAt(moving(spgt_0,spgt_1),106). :- not holdsAt(moving(spgt_1,spgt_0),106).
:- not holdsAt(moving(spgt_0,spgt_1),107). :- not holdsAt(moving(spgt_1,spgt_0),107).
:- not holdsAt(moving(spgt_0,spgt_1),108). :- not holdsAt(moving(spgt_1,spgt_0),108).
:- holdsAt(moving(spgt_0,spgt_1),109). :- holdsAt(moving(spgt_1,spgt_0),109).
:- simpleEvent(T,P,inactive), T>84 , T<108.
:- simpleEvent(T,P,running), T>84 , T<108.

#show simpleEvent/3.