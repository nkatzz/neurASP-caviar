

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
orientation(wk2gt_1,138,96). coords(wk2gt_1,115,53,96).  visible(wk2gt_1,96). orientation(wk2gt_2,141,96). coords(wk2gt_2,98,62,96).  visible(wk2gt_2,96).
orientation(wk2gt_1,139,97). coords(wk2gt_1,115,53,97).  visible(wk2gt_1,97). orientation(wk2gt_2,141,97). coords(wk2gt_2,98,62,97).  visible(wk2gt_2,97).
orientation(wk2gt_1,136,98). coords(wk2gt_1,115,53,98).  visible(wk2gt_1,98). orientation(wk2gt_2,141,98). coords(wk2gt_2,98,62,98).  visible(wk2gt_2,98).
orientation(wk2gt_1,136,99). coords(wk2gt_1,115,52,99).  visible(wk2gt_1,99). orientation(wk2gt_2,141,99). coords(wk2gt_2,98,62,99).  visible(wk2gt_2,99).
orientation(wk2gt_1,135,100). coords(wk2gt_1,115,52,100).  visible(wk2gt_1,100). orientation(wk2gt_2,141,100). coords(wk2gt_2,98,62,100).  visible(wk2gt_2,100).
orientation(wk2gt_1,135,101). coords(wk2gt_1,115,52,101).  visible(wk2gt_1,101). orientation(wk2gt_2,141,101). coords(wk2gt_2,98,62,101).  visible(wk2gt_2,101).
orientation(wk2gt_1,138,102). coords(wk2gt_1,115,52,102).  visible(wk2gt_1,102). orientation(wk2gt_2,141,102). coords(wk2gt_2,98,62,102).  visible(wk2gt_2,102).
orientation(wk2gt_1,139,103). coords(wk2gt_1,115,52,103).  visible(wk2gt_1,103). orientation(wk2gt_2,141,103). coords(wk2gt_2,98,62,103).  visible(wk2gt_2,103).
orientation(wk2gt_1,135,104). coords(wk2gt_1,115,52,104).  visible(wk2gt_1,104). orientation(wk2gt_2,141,104). coords(wk2gt_2,98,62,104).  visible(wk2gt_2,104).
orientation(wk2gt_1,135,105). coords(wk2gt_1,115,52,105).  visible(wk2gt_1,105). orientation(wk2gt_2,141,105). coords(wk2gt_2,98,62,105).  visible(wk2gt_2,105).
orientation(wk2gt_1,134,106). coords(wk2gt_1,115,52,106).  visible(wk2gt_1,106). orientation(wk2gt_2,141,106). coords(wk2gt_2,98,62,106).  visible(wk2gt_2,106).
orientation(wk2gt_1,136,107). coords(wk2gt_1,115,52,107).  visible(wk2gt_1,107). orientation(wk2gt_2,141,107). coords(wk2gt_2,98,62,107).  visible(wk2gt_2,107).
orientation(wk2gt_1,137,108). coords(wk2gt_1,115,52,108).  visible(wk2gt_1,108). orientation(wk2gt_2,141,108). coords(wk2gt_2,98,62,108).  visible(wk2gt_2,108).
orientation(wk2gt_1,134,109). coords(wk2gt_1,115,52,109).  visible(wk2gt_1,109). orientation(wk2gt_2,141,109). coords(wk2gt_2,98,62,109).  visible(wk2gt_2,109).
orientation(wk2gt_1,136,110). coords(wk2gt_1,115,52,110).  visible(wk2gt_1,110). orientation(wk2gt_2,141,110). coords(wk2gt_2,98,62,110).  visible(wk2gt_2,110).
orientation(wk2gt_1,134,111). coords(wk2gt_1,115,52,111).  visible(wk2gt_1,111). orientation(wk2gt_2,141,111). coords(wk2gt_2,98,62,111).  visible(wk2gt_2,111).
orientation(wk2gt_1,136,112). coords(wk2gt_1,115,52,112).  visible(wk2gt_1,112). orientation(wk2gt_2,141,112). coords(wk2gt_2,98,62,112).  visible(wk2gt_2,112).
orientation(wk2gt_1,139,113). coords(wk2gt_1,115,52,113).  visible(wk2gt_1,113). orientation(wk2gt_2,141,113). coords(wk2gt_2,98,62,113).  visible(wk2gt_2,113).
orientation(wk2gt_1,136,114). coords(wk2gt_1,115,52,114).  visible(wk2gt_1,114). orientation(wk2gt_2,141,114). coords(wk2gt_2,98,62,114).  visible(wk2gt_2,114).
orientation(wk2gt_1,135,115). coords(wk2gt_1,115,52,115).  visible(wk2gt_1,115). orientation(wk2gt_2,141,115). coords(wk2gt_2,98,62,115).  visible(wk2gt_2,115).
orientation(wk2gt_1,136,116). coords(wk2gt_1,115,52,116).  visible(wk2gt_1,116). orientation(wk2gt_2,141,116). coords(wk2gt_2,98,62,116).  visible(wk2gt_2,116).
orientation(wk2gt_1,135,117). coords(wk2gt_1,115,52,117).  visible(wk2gt_1,117). orientation(wk2gt_2,141,117). coords(wk2gt_2,98,62,117).  visible(wk2gt_2,117).
orientation(wk2gt_1,136,118). coords(wk2gt_1,115,52,118).  visible(wk2gt_1,118). orientation(wk2gt_2,141,118). coords(wk2gt_2,98,62,118).  visible(wk2gt_2,118).
orientation(wk2gt_1,136,119). coords(wk2gt_1,115,52,119).  visible(wk2gt_1,119). orientation(wk2gt_2,141,119). coords(wk2gt_2,98,62,119).  visible(wk2gt_2,119).
orientation(wk2gt_1,132,120). coords(wk2gt_1,114,52,120).  visible(wk2gt_1,120). orientation(wk2gt_2,141,120). coords(wk2gt_2,98,62,120).  visible(wk2gt_2,120).
orientation(wk2gt_1,132,121). coords(wk2gt_1,114,52,121).  visible(wk2gt_1,121). orientation(wk2gt_2,141,121). coords(wk2gt_2,98,62,121).  visible(wk2gt_2,121).
happensAt(disappear(wk2gt_1),120). happensAt(disappear(wk2gt_2),120).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=96..120.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=96..120.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),96). :- holdsAt(meeting(wk2gt_2,wk2gt_1),96).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),97). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),97).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),98). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),98).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),99). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),99).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),100). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),100).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),101). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),101).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),102). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),102).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),103). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),103).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),104). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),104).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),105). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),105).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),106). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),106).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),107). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),107).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),108). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),108).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),109). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),109).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),110). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),110).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),111). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),111).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),112). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),112).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),113). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),113).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),114). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),114).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),115). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),115).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),116). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),116).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),117). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),117).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),118). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),118).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),119). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),119).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),120). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),120).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),121). :- holdsAt(meeting(wk2gt_2,wk2gt_1),121).
:- simpleEvent(T,P,walking), T>96 , T<120.
:- simpleEvent(T,P,running), T>96 , T<120.

#show simpleEvent/3.