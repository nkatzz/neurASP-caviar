

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
orientation(wk1gt_4,0,99). coords(wk1gt_4,41,193,99).  visible(wk1gt_4,99). orientation(wk1gt_5,0,99). coords(wk1gt_5,35,188,99).  visible(wk1gt_5,99).
orientation(wk1gt_4,0,100). coords(wk1gt_4,41,193,100).  visible(wk1gt_4,100). orientation(wk1gt_5,0,100). coords(wk1gt_5,36,187,100).  visible(wk1gt_5,100).
orientation(wk1gt_4,0,101). coords(wk1gt_4,43,191,101).  visible(wk1gt_4,101). orientation(wk1gt_5,0,101). coords(wk1gt_5,37,186,101).  visible(wk1gt_5,101).
orientation(wk1gt_4,0,102). coords(wk1gt_4,44,190,102).  visible(wk1gt_4,102). orientation(wk1gt_5,0,102). coords(wk1gt_5,37,185,102).  visible(wk1gt_5,102).
orientation(wk1gt_4,0,103). coords(wk1gt_4,44,190,103).  visible(wk1gt_4,103). orientation(wk1gt_5,0,103). coords(wk1gt_5,37,183,103).  visible(wk1gt_5,103).
orientation(wk1gt_4,0,104). coords(wk1gt_4,44,190,104).  visible(wk1gt_4,104). orientation(wk1gt_5,0,104). coords(wk1gt_5,37,183,104).  visible(wk1gt_5,104).
orientation(wk1gt_4,0,105). coords(wk1gt_4,44,189,105).  visible(wk1gt_4,105). orientation(wk1gt_5,0,105). coords(wk1gt_5,37,183,105).  visible(wk1gt_5,105).
orientation(wk1gt_4,0,106). coords(wk1gt_4,44,189,106).  visible(wk1gt_4,106). orientation(wk1gt_5,0,106). coords(wk1gt_5,37,181,106).  visible(wk1gt_5,106).
orientation(wk1gt_4,0,107). coords(wk1gt_4,44,188,107).  visible(wk1gt_4,107). orientation(wk1gt_5,0,107). coords(wk1gt_5,38,181,107).  visible(wk1gt_5,107).
orientation(wk1gt_4,0,108). coords(wk1gt_4,44,187,108).  visible(wk1gt_4,108). orientation(wk1gt_5,0,108). coords(wk1gt_5,38,179,108).  visible(wk1gt_5,108).
orientation(wk1gt_4,0,109). coords(wk1gt_4,44,187,109).  visible(wk1gt_4,109). orientation(wk1gt_5,0,109). coords(wk1gt_5,39,179,109).  visible(wk1gt_5,109).
orientation(wk1gt_4,0,110). coords(wk1gt_4,44,187,110).  visible(wk1gt_4,110). orientation(wk1gt_5,0,110). coords(wk1gt_5,39,179,110).  visible(wk1gt_5,110).
orientation(wk1gt_4,0,111). coords(wk1gt_4,44,187,111).  visible(wk1gt_4,111). orientation(wk1gt_5,0,111). coords(wk1gt_5,39,179,111).  visible(wk1gt_5,111).
orientation(wk1gt_4,0,112). coords(wk1gt_4,44,187,112).  visible(wk1gt_4,112). orientation(wk1gt_5,0,112). coords(wk1gt_5,39,179,112).  visible(wk1gt_5,112).
orientation(wk1gt_4,0,113). coords(wk1gt_4,44,187,113).  visible(wk1gt_4,113). orientation(wk1gt_5,174,113). coords(wk1gt_5,39,178,113).  visible(wk1gt_5,113).
orientation(wk1gt_4,0,114). coords(wk1gt_4,44,187,114).  visible(wk1gt_4,114). orientation(wk1gt_5,174,114). coords(wk1gt_5,40,177,114).  visible(wk1gt_5,114).
orientation(wk1gt_4,0,115). coords(wk1gt_4,44,187,115).  visible(wk1gt_4,115). orientation(wk1gt_5,174,115). coords(wk1gt_5,41,176,115).  visible(wk1gt_5,115).
orientation(wk1gt_4,0,116). coords(wk1gt_4,44,187,116).  visible(wk1gt_4,116). orientation(wk1gt_5,174,116). coords(wk1gt_5,41,176,116).  visible(wk1gt_5,116).
orientation(wk1gt_4,0,117). coords(wk1gt_4,44,187,117).  visible(wk1gt_4,117). orientation(wk1gt_5,169,117). coords(wk1gt_5,41,176,117).  visible(wk1gt_5,117).
orientation(wk1gt_4,0,118). coords(wk1gt_4,44,187,118).  visible(wk1gt_4,118). orientation(wk1gt_5,169,118). coords(wk1gt_5,40,176,118).  visible(wk1gt_5,118).
orientation(wk1gt_4,0,119). coords(wk1gt_4,44,187,119).  visible(wk1gt_4,119). orientation(wk1gt_5,169,119). coords(wk1gt_5,40,176,119).  visible(wk1gt_5,119).
orientation(wk1gt_4,0,120). coords(wk1gt_4,44,187,120).  visible(wk1gt_4,120). orientation(wk1gt_5,169,120). coords(wk1gt_5,40,175,120).  visible(wk1gt_5,120).
orientation(wk1gt_4,0,121). coords(wk1gt_4,45,186,121).  visible(wk1gt_4,121). orientation(wk1gt_5,169,121). coords(wk1gt_5,40,175,121).  visible(wk1gt_5,121).
orientation(wk1gt_4,0,122). coords(wk1gt_4,45,186,122).  visible(wk1gt_4,122). orientation(wk1gt_5,169,122). coords(wk1gt_5,40,174,122).  visible(wk1gt_5,122).
orientation(wk1gt_4,0,123). coords(wk1gt_4,45,186,123).  visible(wk1gt_4,123). orientation(wk1gt_5,169,123). coords(wk1gt_5,40,173,123).  visible(wk1gt_5,123).
orientation(wk1gt_4,0,124). coords(wk1gt_4,45,186,124).  visible(wk1gt_4,124). orientation(wk1gt_5,169,124). coords(wk1gt_5,40,173,124).  visible(wk1gt_5,124).
happensAt(disappear(wk1gt_4),123). happensAt(disappear(wk1gt_5),123).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=99..123.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=99..123.
:- holdsAt(moving(wk1gt_4,wk1gt_5),99). :- holdsAt(moving(wk1gt_5,wk1gt_4),99).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),100). :- not holdsAt(moving(wk1gt_5,wk1gt_4),100).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),101). :- not holdsAt(moving(wk1gt_5,wk1gt_4),101).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),102). :- not holdsAt(moving(wk1gt_5,wk1gt_4),102).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),103). :- not holdsAt(moving(wk1gt_5,wk1gt_4),103).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),104). :- not holdsAt(moving(wk1gt_5,wk1gt_4),104).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),105). :- not holdsAt(moving(wk1gt_5,wk1gt_4),105).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),106). :- not holdsAt(moving(wk1gt_5,wk1gt_4),106).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),107). :- not holdsAt(moving(wk1gt_5,wk1gt_4),107).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),108). :- not holdsAt(moving(wk1gt_5,wk1gt_4),108).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),109). :- not holdsAt(moving(wk1gt_5,wk1gt_4),109).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),110). :- not holdsAt(moving(wk1gt_5,wk1gt_4),110).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),111). :- not holdsAt(moving(wk1gt_5,wk1gt_4),111).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),112). :- not holdsAt(moving(wk1gt_5,wk1gt_4),112).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),113). :- not holdsAt(moving(wk1gt_5,wk1gt_4),113).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),114). :- not holdsAt(moving(wk1gt_5,wk1gt_4),114).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),115). :- not holdsAt(moving(wk1gt_5,wk1gt_4),115).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),116). :- not holdsAt(moving(wk1gt_5,wk1gt_4),116).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),117). :- not holdsAt(moving(wk1gt_5,wk1gt_4),117).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),118). :- not holdsAt(moving(wk1gt_5,wk1gt_4),118).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),119). :- not holdsAt(moving(wk1gt_5,wk1gt_4),119).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),120). :- not holdsAt(moving(wk1gt_5,wk1gt_4),120).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),121). :- not holdsAt(moving(wk1gt_5,wk1gt_4),121).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),122). :- not holdsAt(moving(wk1gt_5,wk1gt_4),122).
:- not holdsAt(moving(wk1gt_4,wk1gt_5),123). :- not holdsAt(moving(wk1gt_5,wk1gt_4),123).
:- holdsAt(moving(wk1gt_4,wk1gt_5),124). :- holdsAt(moving(wk1gt_5,wk1gt_4),124).
:- simpleEvent(T,P,inactive), T>99 , T<123.
:- simpleEvent(T,P,running), T>99 , T<123.

#show simpleEvent/3.