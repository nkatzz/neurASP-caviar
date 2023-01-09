

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
orientation(spgt_0,148,192). coords(spgt_0,113,114,192).  visible(spgt_0,192). orientation(spgt_1,148,192). coords(spgt_1,108,103,192).  visible(spgt_1,192).
orientation(spgt_0,148,193). coords(spgt_0,113,114,193).  visible(spgt_0,193). orientation(spgt_1,148,193). coords(spgt_1,108,103,193).  visible(spgt_1,193).
orientation(spgt_0,148,194). coords(spgt_0,113,114,194).  visible(spgt_0,194). orientation(spgt_1,148,194). coords(spgt_1,108,103,194).  visible(spgt_1,194).
orientation(spgt_0,148,195). coords(spgt_0,113,114,195).  visible(spgt_0,195). orientation(spgt_1,148,195). coords(spgt_1,108,103,195).  visible(spgt_1,195).
orientation(spgt_0,148,196). coords(spgt_0,113,114,196).  visible(spgt_0,196). orientation(spgt_1,148,196). coords(spgt_1,108,103,196).  visible(spgt_1,196).
orientation(spgt_0,148,197). coords(spgt_0,113,114,197).  visible(spgt_0,197). orientation(spgt_1,148,197). coords(spgt_1,108,103,197).  visible(spgt_1,197).
orientation(spgt_0,148,198). coords(spgt_0,113,114,198).  visible(spgt_0,198). orientation(spgt_1,148,198). coords(spgt_1,108,103,198).  visible(spgt_1,198).
orientation(spgt_0,148,199). coords(spgt_0,113,114,199).  visible(spgt_0,199). orientation(spgt_1,148,199). coords(spgt_1,108,103,199).  visible(spgt_1,199).
orientation(spgt_0,148,200). coords(spgt_0,113,114,200).  visible(spgt_0,200). orientation(spgt_1,148,200). coords(spgt_1,108,103,200).  visible(spgt_1,200).
orientation(spgt_0,148,201). coords(spgt_0,113,114,201).  visible(spgt_0,201). orientation(spgt_1,148,201). coords(spgt_1,108,103,201).  visible(spgt_1,201).
orientation(spgt_0,148,202). coords(spgt_0,113,114,202).  visible(spgt_0,202). orientation(spgt_1,148,202). coords(spgt_1,108,103,202).  visible(spgt_1,202).
orientation(spgt_0,148,203). coords(spgt_0,113,114,203).  visible(spgt_0,203). orientation(spgt_1,148,203). coords(spgt_1,108,103,203).  visible(spgt_1,203).
orientation(spgt_0,148,204). coords(spgt_0,113,114,204).  visible(spgt_0,204). orientation(spgt_1,148,204). coords(spgt_1,108,103,204).  visible(spgt_1,204).
orientation(spgt_0,148,205). coords(spgt_0,113,114,205).  visible(spgt_0,205). orientation(spgt_1,148,205). coords(spgt_1,108,103,205).  visible(spgt_1,205).
orientation(spgt_0,148,206). coords(spgt_0,113,114,206).  visible(spgt_0,206). orientation(spgt_1,148,206). coords(spgt_1,108,103,206).  visible(spgt_1,206).
orientation(spgt_0,148,207). coords(spgt_0,113,114,207).  visible(spgt_0,207). orientation(spgt_1,148,207). coords(spgt_1,108,103,207).  visible(spgt_1,207).
orientation(spgt_0,148,208). coords(spgt_0,113,114,208).  visible(spgt_0,208). orientation(spgt_1,148,208). coords(spgt_1,108,103,208).  visible(spgt_1,208).
orientation(spgt_0,148,209). coords(spgt_0,113,114,209).  visible(spgt_0,209). orientation(spgt_1,148,209). coords(spgt_1,108,103,209).  visible(spgt_1,209).
orientation(spgt_0,148,210). coords(spgt_0,113,114,210).  visible(spgt_0,210). orientation(spgt_1,148,210). coords(spgt_1,108,103,210).  visible(spgt_1,210).
orientation(spgt_0,148,211). coords(spgt_0,113,114,211).  visible(spgt_0,211). orientation(spgt_1,148,211). coords(spgt_1,108,103,211).  visible(spgt_1,211).
orientation(spgt_0,148,212). coords(spgt_0,113,114,212).  visible(spgt_0,212). orientation(spgt_1,148,212). coords(spgt_1,108,103,212).  visible(spgt_1,212).
orientation(spgt_0,148,213). coords(spgt_0,113,114,213).  visible(spgt_0,213). orientation(spgt_1,148,213). coords(spgt_1,108,103,213).  visible(spgt_1,213).
orientation(spgt_0,148,214). coords(spgt_0,113,114,214).  visible(spgt_0,214). orientation(spgt_1,148,214). coords(spgt_1,108,103,214).  visible(spgt_1,214).
orientation(spgt_0,148,215). coords(spgt_0,113,114,215).  visible(spgt_0,215). orientation(spgt_1,148,215). coords(spgt_1,108,103,215).  visible(spgt_1,215).
orientation(spgt_0,148,216). coords(spgt_0,113,114,216).  visible(spgt_0,216). orientation(spgt_1,148,216). coords(spgt_1,108,103,216).  visible(spgt_1,216).
orientation(spgt_0,148,217). coords(spgt_0,113,114,217).  visible(spgt_0,217). orientation(spgt_1,148,217). coords(spgt_1,108,103,217).  visible(spgt_1,217).
happensAt(disappear(spgt_0),216). happensAt(disappear(spgt_1),216).
1{simpleEvent(T,spgt_0,active); simpleEvent(T,spgt_0,inactive); simpleEvent(T,spgt_0,walking); simpleEvent(T,spgt_0,running)}1 :- T=192..216.
1{simpleEvent(T,spgt_1,active); simpleEvent(T,spgt_1,inactive); simpleEvent(T,spgt_1,walking); simpleEvent(T,spgt_1,running)}1 :- T=192..216.
:- holdsAt(meeting(spgt_0,spgt_1),192). :- holdsAt(meeting(spgt_1,spgt_0),192).
:- not holdsAt(meeting(spgt_0,spgt_1),193). :- not holdsAt(meeting(spgt_1,spgt_0),193).
:- not holdsAt(meeting(spgt_0,spgt_1),194). :- not holdsAt(meeting(spgt_1,spgt_0),194).
:- not holdsAt(meeting(spgt_0,spgt_1),195). :- not holdsAt(meeting(spgt_1,spgt_0),195).
:- not holdsAt(meeting(spgt_0,spgt_1),196). :- not holdsAt(meeting(spgt_1,spgt_0),196).
:- not holdsAt(meeting(spgt_0,spgt_1),197). :- not holdsAt(meeting(spgt_1,spgt_0),197).
:- not holdsAt(meeting(spgt_0,spgt_1),198). :- not holdsAt(meeting(spgt_1,spgt_0),198).
:- not holdsAt(meeting(spgt_0,spgt_1),199). :- not holdsAt(meeting(spgt_1,spgt_0),199).
:- not holdsAt(meeting(spgt_0,spgt_1),200). :- not holdsAt(meeting(spgt_1,spgt_0),200).
:- not holdsAt(meeting(spgt_0,spgt_1),201). :- not holdsAt(meeting(spgt_1,spgt_0),201).
:- not holdsAt(meeting(spgt_0,spgt_1),202). :- not holdsAt(meeting(spgt_1,spgt_0),202).
:- not holdsAt(meeting(spgt_0,spgt_1),203). :- not holdsAt(meeting(spgt_1,spgt_0),203).
:- not holdsAt(meeting(spgt_0,spgt_1),204). :- not holdsAt(meeting(spgt_1,spgt_0),204).
:- not holdsAt(meeting(spgt_0,spgt_1),205). :- not holdsAt(meeting(spgt_1,spgt_0),205).
:- not holdsAt(meeting(spgt_0,spgt_1),206). :- not holdsAt(meeting(spgt_1,spgt_0),206).
:- not holdsAt(meeting(spgt_0,spgt_1),207). :- not holdsAt(meeting(spgt_1,spgt_0),207).
:- not holdsAt(meeting(spgt_0,spgt_1),208). :- not holdsAt(meeting(spgt_1,spgt_0),208).
:- not holdsAt(meeting(spgt_0,spgt_1),209). :- not holdsAt(meeting(spgt_1,spgt_0),209).
:- not holdsAt(meeting(spgt_0,spgt_1),210). :- not holdsAt(meeting(spgt_1,spgt_0),210).
:- not holdsAt(meeting(spgt_0,spgt_1),211). :- not holdsAt(meeting(spgt_1,spgt_0),211).
:- not holdsAt(meeting(spgt_0,spgt_1),212). :- not holdsAt(meeting(spgt_1,spgt_0),212).
:- not holdsAt(meeting(spgt_0,spgt_1),213). :- not holdsAt(meeting(spgt_1,spgt_0),213).
:- not holdsAt(meeting(spgt_0,spgt_1),214). :- not holdsAt(meeting(spgt_1,spgt_0),214).
:- not holdsAt(meeting(spgt_0,spgt_1),215). :- not holdsAt(meeting(spgt_1,spgt_0),215).
:- not holdsAt(meeting(spgt_0,spgt_1),216). :- not holdsAt(meeting(spgt_1,spgt_0),216).
:- holdsAt(meeting(spgt_0,spgt_1),217). :- holdsAt(meeting(spgt_1,spgt_0),217).
:- simpleEvent(T,P,walking), T>192 , T<216.
:- simpleEvent(T,P,running), T>192 , T<216.

#show simpleEvent/3.