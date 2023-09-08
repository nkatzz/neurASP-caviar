

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

	person(mws1gt_5;mws1gt_6).
orientation(mws1gt_5,145,192). coords(mws1gt_5,53,99,192).  visible(mws1gt_5,192). orientation(mws1gt_6,140,192). coords(mws1gt_6,66,81,192).  visible(mws1gt_6,192).
orientation(mws1gt_5,145,193). coords(mws1gt_5,53,99,193).  visible(mws1gt_5,193). orientation(mws1gt_6,140,193). coords(mws1gt_6,66,82,193).  visible(mws1gt_6,193).
orientation(mws1gt_5,145,194). coords(mws1gt_5,49,99,194).  visible(mws1gt_5,194). orientation(mws1gt_6,140,194). coords(mws1gt_6,66,83,194).  visible(mws1gt_6,194).
orientation(mws1gt_5,145,195). coords(mws1gt_5,49,99,195).  visible(mws1gt_5,195). orientation(mws1gt_6,140,195). coords(mws1gt_6,66,84,195).  visible(mws1gt_6,195).
orientation(mws1gt_5,145,196). coords(mws1gt_5,49,99,196).  visible(mws1gt_5,196). orientation(mws1gt_6,140,196). coords(mws1gt_6,65,84,196).  visible(mws1gt_6,196).
orientation(mws1gt_5,158,197). coords(mws1gt_5,48,100,197).  visible(mws1gt_5,197). orientation(mws1gt_6,140,197). coords(mws1gt_6,64,84,197).  visible(mws1gt_6,197).
orientation(mws1gt_5,158,198). coords(mws1gt_5,48,100,198).  visible(mws1gt_5,198). orientation(mws1gt_6,140,198). coords(mws1gt_6,64,84,198).  visible(mws1gt_6,198).
orientation(mws1gt_5,161,199). coords(mws1gt_5,48,102,199).  visible(mws1gt_5,199). orientation(mws1gt_6,140,199). coords(mws1gt_6,64,85,199).  visible(mws1gt_6,199).
orientation(mws1gt_5,161,200). coords(mws1gt_5,48,102,200).  visible(mws1gt_5,200). orientation(mws1gt_6,140,200). coords(mws1gt_6,63,85,200).  visible(mws1gt_6,200).
orientation(mws1gt_5,161,201). coords(mws1gt_5,48,102,201).  visible(mws1gt_5,201). orientation(mws1gt_6,140,201). coords(mws1gt_6,62,86,201).  visible(mws1gt_6,201).
orientation(mws1gt_5,161,202). coords(mws1gt_5,47,103,202).  visible(mws1gt_5,202). orientation(mws1gt_6,140,202). coords(mws1gt_6,61,86,202).  visible(mws1gt_6,202).
orientation(mws1gt_5,161,203). coords(mws1gt_5,46,104,203).  visible(mws1gt_5,203). orientation(mws1gt_6,140,203). coords(mws1gt_6,60,87,203).  visible(mws1gt_6,203).
orientation(mws1gt_5,161,204). coords(mws1gt_5,45,105,204).  visible(mws1gt_5,204). orientation(mws1gt_6,140,204). coords(mws1gt_6,60,88,204).  visible(mws1gt_6,204).
orientation(mws1gt_5,161,205). coords(mws1gt_5,45,107,205).  visible(mws1gt_5,205). orientation(mws1gt_6,146,205). coords(mws1gt_6,60,90,205).  visible(mws1gt_6,205).
orientation(mws1gt_5,161,206). coords(mws1gt_5,45,107,206).  visible(mws1gt_5,206). orientation(mws1gt_6,146,206). coords(mws1gt_6,59,91,206).  visible(mws1gt_6,206).
orientation(mws1gt_5,161,207). coords(mws1gt_5,45,107,207).  visible(mws1gt_5,207). orientation(mws1gt_6,146,207). coords(mws1gt_6,59,92,207).  visible(mws1gt_6,207).
orientation(mws1gt_5,161,208). coords(mws1gt_5,45,108,208).  visible(mws1gt_5,208). orientation(mws1gt_6,146,208). coords(mws1gt_6,59,92,208).  visible(mws1gt_6,208).
orientation(mws1gt_5,161,209). coords(mws1gt_5,45,109,209).  visible(mws1gt_5,209). orientation(mws1gt_6,146,209). coords(mws1gt_6,59,94,209).  visible(mws1gt_6,209).
orientation(mws1gt_5,161,210). coords(mws1gt_5,45,109,210).  visible(mws1gt_5,210). orientation(mws1gt_6,146,210). coords(mws1gt_6,59,94,210).  visible(mws1gt_6,210).
orientation(mws1gt_5,161,211). coords(mws1gt_5,45,109,211).  visible(mws1gt_5,211). orientation(mws1gt_6,146,211). coords(mws1gt_6,59,94,211).  visible(mws1gt_6,211).
orientation(mws1gt_5,161,212). coords(mws1gt_5,45,110,212).  visible(mws1gt_5,212). orientation(mws1gt_6,146,212). coords(mws1gt_6,59,95,212).  visible(mws1gt_6,212).
orientation(mws1gt_5,161,213). coords(mws1gt_5,45,111,213).  visible(mws1gt_5,213). orientation(mws1gt_6,146,213). coords(mws1gt_6,58,95,213).  visible(mws1gt_6,213).
orientation(mws1gt_5,161,214). coords(mws1gt_5,45,113,214).  visible(mws1gt_5,214). orientation(mws1gt_6,146,214). coords(mws1gt_6,57,95,214).  visible(mws1gt_6,214).
orientation(mws1gt_5,161,215). coords(mws1gt_5,46,116,215).  visible(mws1gt_5,215). orientation(mws1gt_6,146,215). coords(mws1gt_6,56,96,215).  visible(mws1gt_6,215).
orientation(mws1gt_5,161,216). coords(mws1gt_5,46,117,216).  visible(mws1gt_5,216). orientation(mws1gt_6,146,216). coords(mws1gt_6,55,96,216).  visible(mws1gt_6,216).
orientation(mws1gt_5,161,217). coords(mws1gt_5,46,117,217).  visible(mws1gt_5,217). orientation(mws1gt_6,146,217). coords(mws1gt_6,55,96,217).  visible(mws1gt_6,217).
happensAt(disappear(mws1gt_5),216). happensAt(disappear(mws1gt_6),216).
1{simpleEvent(T,mws1gt_5,active); simpleEvent(T,mws1gt_5,inactive); simpleEvent(T,mws1gt_5,walking); simpleEvent(T,mws1gt_5,running)}1 :- T=192..216.
1{simpleEvent(T,mws1gt_6,active); simpleEvent(T,mws1gt_6,inactive); simpleEvent(T,mws1gt_6,walking); simpleEvent(T,mws1gt_6,running)}1 :- T=192..216.
:- holdsAt(moving(mws1gt_5,mws1gt_6),192). :- holdsAt(moving(mws1gt_6,mws1gt_5),192).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),193). :- not holdsAt(moving(mws1gt_6,mws1gt_5),193).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),194). :- not holdsAt(moving(mws1gt_6,mws1gt_5),194).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),195). :- not holdsAt(moving(mws1gt_6,mws1gt_5),195).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),196). :- not holdsAt(moving(mws1gt_6,mws1gt_5),196).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),197). :- not holdsAt(moving(mws1gt_6,mws1gt_5),197).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),198). :- not holdsAt(moving(mws1gt_6,mws1gt_5),198).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),199). :- not holdsAt(moving(mws1gt_6,mws1gt_5),199).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),200). :- not holdsAt(moving(mws1gt_6,mws1gt_5),200).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),201). :- not holdsAt(moving(mws1gt_6,mws1gt_5),201).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),202). :- not holdsAt(moving(mws1gt_6,mws1gt_5),202).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),203). :- not holdsAt(moving(mws1gt_6,mws1gt_5),203).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),204). :- not holdsAt(moving(mws1gt_6,mws1gt_5),204).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),205). :- not holdsAt(moving(mws1gt_6,mws1gt_5),205).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),206). :- not holdsAt(moving(mws1gt_6,mws1gt_5),206).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),207). :- not holdsAt(moving(mws1gt_6,mws1gt_5),207).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),208). :- not holdsAt(moving(mws1gt_6,mws1gt_5),208).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),209). :- not holdsAt(moving(mws1gt_6,mws1gt_5),209).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),210). :- not holdsAt(moving(mws1gt_6,mws1gt_5),210).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),211). :- not holdsAt(moving(mws1gt_6,mws1gt_5),211).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),212). :- not holdsAt(moving(mws1gt_6,mws1gt_5),212).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),213). :- not holdsAt(moving(mws1gt_6,mws1gt_5),213).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),214). :- not holdsAt(moving(mws1gt_6,mws1gt_5),214).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),215). :- not holdsAt(moving(mws1gt_6,mws1gt_5),215).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),216). :- not holdsAt(moving(mws1gt_6,mws1gt_5),216).
:- holdsAt(moving(mws1gt_5,mws1gt_6),217). :- holdsAt(moving(mws1gt_6,mws1gt_5),217).
:- simpleEvent(T,P,inactive), T>192 , T<216.
:- simpleEvent(T,P,running), T>192 , T<216.

#show simpleEvent/3.