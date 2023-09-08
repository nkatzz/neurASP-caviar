

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

	person(mws1gt_1;mws1gt_3).
orientation(mws1gt_1,155,188). coords(mws1gt_1,147,136,188).  visible(mws1gt_1,188). orientation(mws1gt_3,154,188). coords(mws1gt_3,153,153,188).  visible(mws1gt_3,188).
orientation(mws1gt_1,155,189). coords(mws1gt_1,146,136,189).  visible(mws1gt_1,189). orientation(mws1gt_3,154,189). coords(mws1gt_3,153,154,189).  visible(mws1gt_3,189).
orientation(mws1gt_1,155,190). coords(mws1gt_1,145,137,190).  visible(mws1gt_1,190). orientation(mws1gt_3,154,190). coords(mws1gt_3,152,154,190).  visible(mws1gt_3,190).
orientation(mws1gt_1,155,191). coords(mws1gt_1,144,137,191).  visible(mws1gt_1,191). orientation(mws1gt_3,158,191). coords(mws1gt_3,152,154,191).  visible(mws1gt_3,191).
orientation(mws1gt_1,155,192). coords(mws1gt_1,144,136,192).  visible(mws1gt_1,192). orientation(mws1gt_3,160,192). coords(mws1gt_3,151,154,192).  visible(mws1gt_3,192).
orientation(mws1gt_1,155,193). coords(mws1gt_1,144,136,193).  visible(mws1gt_1,193). orientation(mws1gt_3,163,193). coords(mws1gt_3,150,154,193).  visible(mws1gt_3,193).
orientation(mws1gt_1,155,194). coords(mws1gt_1,143,136,194).  visible(mws1gt_1,194). orientation(mws1gt_3,162,194). coords(mws1gt_3,149,154,194).  visible(mws1gt_3,194).
orientation(mws1gt_1,155,195). coords(mws1gt_1,143,135,195).  visible(mws1gt_1,195). orientation(mws1gt_3,161,195). coords(mws1gt_3,147,154,195).  visible(mws1gt_3,195).
orientation(mws1gt_1,155,196). coords(mws1gt_1,143,135,196).  visible(mws1gt_1,196). orientation(mws1gt_3,162,196). coords(mws1gt_3,146,154,196).  visible(mws1gt_3,196).
orientation(mws1gt_1,155,197). coords(mws1gt_1,142,135,197).  visible(mws1gt_1,197). orientation(mws1gt_3,162,197). coords(mws1gt_3,144,154,197).  visible(mws1gt_3,197).
orientation(mws1gt_1,155,198). coords(mws1gt_1,142,134,198).  visible(mws1gt_1,198). orientation(mws1gt_3,162,198). coords(mws1gt_3,143,154,198).  visible(mws1gt_3,198).
orientation(mws1gt_1,155,199). coords(mws1gt_1,142,134,199).  visible(mws1gt_1,199). orientation(mws1gt_3,162,199). coords(mws1gt_3,141,154,199).  visible(mws1gt_3,199).
orientation(mws1gt_1,155,200). coords(mws1gt_1,140,133,200).  visible(mws1gt_1,200). orientation(mws1gt_3,162,200). coords(mws1gt_3,140,154,200).  visible(mws1gt_3,200).
orientation(mws1gt_1,155,201). coords(mws1gt_1,139,133,201).  visible(mws1gt_1,201). orientation(mws1gt_3,162,201). coords(mws1gt_3,140,154,201).  visible(mws1gt_3,201).
orientation(mws1gt_1,155,202). coords(mws1gt_1,139,133,202).  visible(mws1gt_1,202). orientation(mws1gt_3,162,202). coords(mws1gt_3,139,155,202).  visible(mws1gt_3,202).
orientation(mws1gt_1,155,203). coords(mws1gt_1,138,133,203).  visible(mws1gt_1,203). orientation(mws1gt_3,162,203). coords(mws1gt_3,139,155,203).  visible(mws1gt_3,203).
orientation(mws1gt_1,155,204). coords(mws1gt_1,138,133,204).  visible(mws1gt_1,204). orientation(mws1gt_3,162,204). coords(mws1gt_3,139,155,204).  visible(mws1gt_3,204).
orientation(mws1gt_1,155,205). coords(mws1gt_1,138,133,205).  visible(mws1gt_1,205). orientation(mws1gt_3,162,205). coords(mws1gt_3,139,155,205).  visible(mws1gt_3,205).
orientation(mws1gt_1,155,206). coords(mws1gt_1,137,132,206).  visible(mws1gt_1,206). orientation(mws1gt_3,162,206). coords(mws1gt_3,138,155,206).  visible(mws1gt_3,206).
orientation(mws1gt_1,155,207). coords(mws1gt_1,137,132,207).  visible(mws1gt_1,207). orientation(mws1gt_3,162,207). coords(mws1gt_3,138,155,207).  visible(mws1gt_3,207).
orientation(mws1gt_1,155,208). coords(mws1gt_1,137,132,208).  visible(mws1gt_1,208). orientation(mws1gt_3,162,208). coords(mws1gt_3,137,155,208).  visible(mws1gt_3,208).
orientation(mws1gt_1,155,209). coords(mws1gt_1,136,132,209).  visible(mws1gt_1,209). orientation(mws1gt_3,162,209). coords(mws1gt_3,135,155,209).  visible(mws1gt_3,209).
orientation(mws1gt_1,155,210). coords(mws1gt_1,135,132,210).  visible(mws1gt_1,210). orientation(mws1gt_3,162,210). coords(mws1gt_3,134,155,210).  visible(mws1gt_3,210).
orientation(mws1gt_1,155,211). coords(mws1gt_1,134,132,211).  visible(mws1gt_1,211). orientation(mws1gt_3,162,211). coords(mws1gt_3,132,155,211).  visible(mws1gt_3,211).
orientation(mws1gt_1,155,212). coords(mws1gt_1,134,132,212).  visible(mws1gt_1,212). orientation(mws1gt_3,162,212). coords(mws1gt_3,130,155,212).  visible(mws1gt_3,212).
orientation(mws1gt_1,155,213). coords(mws1gt_1,134,132,213).  visible(mws1gt_1,213). orientation(mws1gt_3,162,213). coords(mws1gt_3,130,155,213).  visible(mws1gt_3,213).
happensAt(disappear(mws1gt_1),212). happensAt(disappear(mws1gt_3),212).
1{simpleEvent(T,mws1gt_1,active); simpleEvent(T,mws1gt_1,inactive); simpleEvent(T,mws1gt_1,walking); simpleEvent(T,mws1gt_1,running)}1 :- T=188..212.
1{simpleEvent(T,mws1gt_3,active); simpleEvent(T,mws1gt_3,inactive); simpleEvent(T,mws1gt_3,walking); simpleEvent(T,mws1gt_3,running)}1 :- T=188..212.
:- holdsAt(moving(mws1gt_1,mws1gt_3),188). :- holdsAt(moving(mws1gt_3,mws1gt_1),188).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),189). :- not holdsAt(moving(mws1gt_3,mws1gt_1),189).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),190). :- not holdsAt(moving(mws1gt_3,mws1gt_1),190).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),191). :- not holdsAt(moving(mws1gt_3,mws1gt_1),191).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),192). :- not holdsAt(moving(mws1gt_3,mws1gt_1),192).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),193). :- not holdsAt(moving(mws1gt_3,mws1gt_1),193).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),194). :- not holdsAt(moving(mws1gt_3,mws1gt_1),194).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),195). :- not holdsAt(moving(mws1gt_3,mws1gt_1),195).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),196). :- not holdsAt(moving(mws1gt_3,mws1gt_1),196).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),197). :- not holdsAt(moving(mws1gt_3,mws1gt_1),197).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),198). :- not holdsAt(moving(mws1gt_3,mws1gt_1),198).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),199). :- not holdsAt(moving(mws1gt_3,mws1gt_1),199).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),200). :- not holdsAt(moving(mws1gt_3,mws1gt_1),200).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),201). :- not holdsAt(moving(mws1gt_3,mws1gt_1),201).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),202). :- not holdsAt(moving(mws1gt_3,mws1gt_1),202).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),203). :- not holdsAt(moving(mws1gt_3,mws1gt_1),203).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),204). :- not holdsAt(moving(mws1gt_3,mws1gt_1),204).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),205). :- not holdsAt(moving(mws1gt_3,mws1gt_1),205).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),206). :- not holdsAt(moving(mws1gt_3,mws1gt_1),206).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),207). :- not holdsAt(moving(mws1gt_3,mws1gt_1),207).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),208). :- not holdsAt(moving(mws1gt_3,mws1gt_1),208).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),209). :- not holdsAt(moving(mws1gt_3,mws1gt_1),209).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),210). :- not holdsAt(moving(mws1gt_3,mws1gt_1),210).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),211). :- not holdsAt(moving(mws1gt_3,mws1gt_1),211).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),212). :- not holdsAt(moving(mws1gt_3,mws1gt_1),212).
:- holdsAt(moving(mws1gt_1,mws1gt_3),213). :- holdsAt(moving(mws1gt_3,mws1gt_1),213).
:- simpleEvent(T,P,inactive), T>188 , T<212.
:- simpleEvent(T,P,running), T>188 , T<212.

#show simpleEvent/3.