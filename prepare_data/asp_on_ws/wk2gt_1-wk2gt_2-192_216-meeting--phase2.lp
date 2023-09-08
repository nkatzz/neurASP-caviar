

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
orientation(wk2gt_1,129,192). coords(wk2gt_1,115,53,192).  visible(wk2gt_1,192). orientation(wk2gt_2,141,192). coords(wk2gt_2,98,62,192).  visible(wk2gt_2,192).
orientation(wk2gt_1,129,193). coords(wk2gt_1,115,53,193).  visible(wk2gt_1,193). orientation(wk2gt_2,141,193). coords(wk2gt_2,98,62,193).  visible(wk2gt_2,193).
orientation(wk2gt_1,129,194). coords(wk2gt_1,115,53,194).  visible(wk2gt_1,194). orientation(wk2gt_2,141,194). coords(wk2gt_2,98,62,194).  visible(wk2gt_2,194).
orientation(wk2gt_1,129,195). coords(wk2gt_1,115,53,195).  visible(wk2gt_1,195). orientation(wk2gt_2,141,195). coords(wk2gt_2,98,62,195).  visible(wk2gt_2,195).
orientation(wk2gt_1,129,196). coords(wk2gt_1,115,53,196).  visible(wk2gt_1,196). orientation(wk2gt_2,141,196). coords(wk2gt_2,98,62,196).  visible(wk2gt_2,196).
orientation(wk2gt_1,129,197). coords(wk2gt_1,115,53,197).  visible(wk2gt_1,197). orientation(wk2gt_2,141,197). coords(wk2gt_2,98,62,197).  visible(wk2gt_2,197).
orientation(wk2gt_1,129,198). coords(wk2gt_1,115,53,198).  visible(wk2gt_1,198). orientation(wk2gt_2,141,198). coords(wk2gt_2,98,62,198).  visible(wk2gt_2,198).
orientation(wk2gt_1,129,199). coords(wk2gt_1,115,53,199).  visible(wk2gt_1,199). orientation(wk2gt_2,141,199). coords(wk2gt_2,98,62,199).  visible(wk2gt_2,199).
orientation(wk2gt_1,129,200). coords(wk2gt_1,115,53,200).  visible(wk2gt_1,200). orientation(wk2gt_2,141,200). coords(wk2gt_2,98,62,200).  visible(wk2gt_2,200).
orientation(wk2gt_1,129,201). coords(wk2gt_1,115,53,201).  visible(wk2gt_1,201). orientation(wk2gt_2,141,201). coords(wk2gt_2,98,62,201).  visible(wk2gt_2,201).
orientation(wk2gt_1,129,202). coords(wk2gt_1,115,53,202).  visible(wk2gt_1,202). orientation(wk2gt_2,141,202). coords(wk2gt_2,98,62,202).  visible(wk2gt_2,202).
orientation(wk2gt_1,129,203). coords(wk2gt_1,115,53,203).  visible(wk2gt_1,203). orientation(wk2gt_2,141,203). coords(wk2gt_2,98,62,203).  visible(wk2gt_2,203).
orientation(wk2gt_1,129,204). coords(wk2gt_1,115,53,204).  visible(wk2gt_1,204). orientation(wk2gt_2,141,204). coords(wk2gt_2,98,62,204).  visible(wk2gt_2,204).
orientation(wk2gt_1,129,205). coords(wk2gt_1,115,53,205).  visible(wk2gt_1,205). orientation(wk2gt_2,141,205). coords(wk2gt_2,98,62,205).  visible(wk2gt_2,205).
orientation(wk2gt_1,129,206). coords(wk2gt_1,115,53,206).  visible(wk2gt_1,206). orientation(wk2gt_2,141,206). coords(wk2gt_2,98,62,206).  visible(wk2gt_2,206).
orientation(wk2gt_1,129,207). coords(wk2gt_1,115,53,207).  visible(wk2gt_1,207). orientation(wk2gt_2,141,207). coords(wk2gt_2,98,62,207).  visible(wk2gt_2,207).
orientation(wk2gt_1,129,208). coords(wk2gt_1,115,53,208).  visible(wk2gt_1,208). orientation(wk2gt_2,141,208). coords(wk2gt_2,98,62,208).  visible(wk2gt_2,208).
orientation(wk2gt_1,129,209). coords(wk2gt_1,115,53,209).  visible(wk2gt_1,209). orientation(wk2gt_2,141,209). coords(wk2gt_2,98,62,209).  visible(wk2gt_2,209).
orientation(wk2gt_1,129,210). coords(wk2gt_1,115,53,210).  visible(wk2gt_1,210). orientation(wk2gt_2,141,210). coords(wk2gt_2,98,62,210).  visible(wk2gt_2,210).
orientation(wk2gt_1,129,211). coords(wk2gt_1,115,53,211).  visible(wk2gt_1,211). orientation(wk2gt_2,141,211). coords(wk2gt_2,98,62,211).  visible(wk2gt_2,211).
orientation(wk2gt_1,129,212). coords(wk2gt_1,115,53,212).  visible(wk2gt_1,212). orientation(wk2gt_2,141,212). coords(wk2gt_2,98,62,212).  visible(wk2gt_2,212).
orientation(wk2gt_1,129,213). coords(wk2gt_1,115,53,213).  visible(wk2gt_1,213). orientation(wk2gt_2,141,213). coords(wk2gt_2,98,62,213).  visible(wk2gt_2,213).
orientation(wk2gt_1,129,214). coords(wk2gt_1,115,53,214).  visible(wk2gt_1,214). orientation(wk2gt_2,141,214). coords(wk2gt_2,98,62,214).  visible(wk2gt_2,214).
orientation(wk2gt_1,129,215). coords(wk2gt_1,115,53,215).  visible(wk2gt_1,215). orientation(wk2gt_2,141,215). coords(wk2gt_2,98,62,215).  visible(wk2gt_2,215).
orientation(wk2gt_1,129,216). coords(wk2gt_1,115,53,216).  visible(wk2gt_1,216). orientation(wk2gt_2,141,216). coords(wk2gt_2,98,62,216).  visible(wk2gt_2,216).
orientation(wk2gt_1,129,217). coords(wk2gt_1,115,53,217).  visible(wk2gt_1,217). orientation(wk2gt_2,141,217). coords(wk2gt_2,98,62,217).  visible(wk2gt_2,217).
happensAt(disappear(wk2gt_1),216). happensAt(disappear(wk2gt_2),216).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=192..216.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=192..216.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),192). :- holdsAt(meeting(wk2gt_2,wk2gt_1),192).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),193). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),193).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),194). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),194).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),195). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),195).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),196). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),196).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),197). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),197).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),198). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),198).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),199). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),199).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),200). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),200).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),201). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),201).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),202). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),202).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),203). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),203).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),204). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),204).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),205). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),205).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),206). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),206).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),207). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),207).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),208). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),208).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),209). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),209).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),210). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),210).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),211). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),211).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),212). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),212).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),213). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),213).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),214). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),214).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),215). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),215).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),216). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),216).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),217). :- holdsAt(meeting(wk2gt_2,wk2gt_1),217).
:- simpleEvent(T,P,walking), T>192 , T<216.
:- simpleEvent(T,P,running), T>192 , T<216.

#show simpleEvent/3.