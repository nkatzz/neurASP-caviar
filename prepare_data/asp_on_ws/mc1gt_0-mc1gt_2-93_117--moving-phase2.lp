

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
orientation(mc1gt_0,148,106). coords(mc1gt_0,80,98,106).  visible(mc1gt_0,106). orientation(mc1gt_2,144,106). coords(mc1gt_2,76,72,106).  visible(mc1gt_2,106).
orientation(mc1gt_0,148,107). coords(mc1gt_0,80,99,107).  visible(mc1gt_0,107). orientation(mc1gt_2,144,107). coords(mc1gt_2,76,73,107).  visible(mc1gt_2,107).
orientation(mc1gt_0,148,108). coords(mc1gt_0,80,99,108).  visible(mc1gt_0,108). orientation(mc1gt_2,144,108). coords(mc1gt_2,76,73,108).  visible(mc1gt_2,108).
orientation(mc1gt_0,147,109). coords(mc1gt_0,81,100,109).  visible(mc1gt_0,109). orientation(mc1gt_2,144,109). coords(mc1gt_2,76,74,109).  visible(mc1gt_2,109).
orientation(mc1gt_0,147,110). coords(mc1gt_0,81,101,110).  visible(mc1gt_0,110). orientation(mc1gt_2,144,110). coords(mc1gt_2,76,74,110).  visible(mc1gt_2,110).
orientation(mc1gt_0,147,111). coords(mc1gt_0,82,101,111).  visible(mc1gt_0,111). orientation(mc1gt_2,144,111). coords(mc1gt_2,77,74,111).  visible(mc1gt_2,111).
orientation(mc1gt_0,147,112). coords(mc1gt_0,82,101,112).  visible(mc1gt_0,112). orientation(mc1gt_2,144,112). coords(mc1gt_2,77,75,112).  visible(mc1gt_2,112).
orientation(mc1gt_0,147,113). coords(mc1gt_0,82,101,113).  visible(mc1gt_0,113). orientation(mc1gt_2,144,113). coords(mc1gt_2,76,75,113).  visible(mc1gt_2,113).
orientation(mc1gt_0,147,114). coords(mc1gt_0,82,101,114).  visible(mc1gt_0,114). orientation(mc1gt_2,144,114). coords(mc1gt_2,76,76,114).  visible(mc1gt_2,114).
orientation(mc1gt_0,147,115). coords(mc1gt_0,82,101,115).  visible(mc1gt_0,115). orientation(mc1gt_2,144,115). coords(mc1gt_2,76,77,115).  visible(mc1gt_2,115).
orientation(mc1gt_0,147,116). coords(mc1gt_0,82,101,116).  visible(mc1gt_0,116). orientation(mc1gt_2,144,116). coords(mc1gt_2,77,78,116).  visible(mc1gt_2,116).
orientation(mc1gt_0,147,117). coords(mc1gt_0,82,101,117).  visible(mc1gt_0,117). orientation(mc1gt_2,144,117). coords(mc1gt_2,77,78,117).  visible(mc1gt_2,117).
orientation(mc1gt_0,147,118). coords(mc1gt_0,82,101,118).  visible(mc1gt_0,118). orientation(mc1gt_2,144,118). coords(mc1gt_2,77,78,118).  visible(mc1gt_2,118).
happensAt(disappear(mc1gt_0),117). happensAt(disappear(mc1gt_2),117).
1{simpleEvent(T,mc1gt_0,active); simpleEvent(T,mc1gt_0,inactive); simpleEvent(T,mc1gt_0,walking); simpleEvent(T,mc1gt_0,running)}1 :- T=93..117.
1{simpleEvent(T,mc1gt_2,active); simpleEvent(T,mc1gt_2,inactive); simpleEvent(T,mc1gt_2,walking); simpleEvent(T,mc1gt_2,running)}1 :- T=93..117.
:- holdsAt(moving(mc1gt_0,mc1gt_2),93). :- holdsAt(moving(mc1gt_2,mc1gt_0),93).
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
:- not holdsAt(moving(mc1gt_0,mc1gt_2),106). :- not holdsAt(moving(mc1gt_2,mc1gt_0),106).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),107). :- not holdsAt(moving(mc1gt_2,mc1gt_0),107).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),108). :- not holdsAt(moving(mc1gt_2,mc1gt_0),108).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),109). :- not holdsAt(moving(mc1gt_2,mc1gt_0),109).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),110). :- not holdsAt(moving(mc1gt_2,mc1gt_0),110).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),111). :- not holdsAt(moving(mc1gt_2,mc1gt_0),111).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),112). :- not holdsAt(moving(mc1gt_2,mc1gt_0),112).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),113). :- not holdsAt(moving(mc1gt_2,mc1gt_0),113).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),114). :- not holdsAt(moving(mc1gt_2,mc1gt_0),114).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),115). :- not holdsAt(moving(mc1gt_2,mc1gt_0),115).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),116). :- not holdsAt(moving(mc1gt_2,mc1gt_0),116).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),117). :- not holdsAt(moving(mc1gt_2,mc1gt_0),117).
:- holdsAt(moving(mc1gt_0,mc1gt_2),118). :- holdsAt(moving(mc1gt_2,mc1gt_0),118).
:- simpleEvent(T,P,inactive), T>93 , T<117.
:- simpleEvent(T,P,running), T>93 , T<117.

#show simpleEvent/3.