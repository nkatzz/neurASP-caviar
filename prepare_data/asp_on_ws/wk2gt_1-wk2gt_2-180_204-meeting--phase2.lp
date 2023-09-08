

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
orientation(wk2gt_1,129,180). coords(wk2gt_1,115,53,180).  visible(wk2gt_1,180). orientation(wk2gt_2,141,180). coords(wk2gt_2,98,62,180).  visible(wk2gt_2,180).
orientation(wk2gt_1,129,181). coords(wk2gt_1,115,53,181).  visible(wk2gt_1,181). orientation(wk2gt_2,141,181). coords(wk2gt_2,98,62,181).  visible(wk2gt_2,181).
orientation(wk2gt_1,129,182). coords(wk2gt_1,115,53,182).  visible(wk2gt_1,182). orientation(wk2gt_2,141,182). coords(wk2gt_2,98,62,182).  visible(wk2gt_2,182).
orientation(wk2gt_1,129,183). coords(wk2gt_1,115,53,183).  visible(wk2gt_1,183). orientation(wk2gt_2,141,183). coords(wk2gt_2,98,62,183).  visible(wk2gt_2,183).
orientation(wk2gt_1,129,184). coords(wk2gt_1,115,53,184).  visible(wk2gt_1,184). orientation(wk2gt_2,141,184). coords(wk2gt_2,98,62,184).  visible(wk2gt_2,184).
orientation(wk2gt_1,129,185). coords(wk2gt_1,115,53,185).  visible(wk2gt_1,185). orientation(wk2gt_2,141,185). coords(wk2gt_2,98,62,185).  visible(wk2gt_2,185).
orientation(wk2gt_1,129,186). coords(wk2gt_1,115,53,186).  visible(wk2gt_1,186). orientation(wk2gt_2,141,186). coords(wk2gt_2,98,62,186).  visible(wk2gt_2,186).
orientation(wk2gt_1,129,187). coords(wk2gt_1,115,53,187).  visible(wk2gt_1,187). orientation(wk2gt_2,141,187). coords(wk2gt_2,98,62,187).  visible(wk2gt_2,187).
orientation(wk2gt_1,129,188). coords(wk2gt_1,115,53,188).  visible(wk2gt_1,188). orientation(wk2gt_2,141,188). coords(wk2gt_2,98,62,188).  visible(wk2gt_2,188).
orientation(wk2gt_1,129,189). coords(wk2gt_1,115,53,189).  visible(wk2gt_1,189). orientation(wk2gt_2,141,189). coords(wk2gt_2,98,62,189).  visible(wk2gt_2,189).
orientation(wk2gt_1,129,190). coords(wk2gt_1,115,53,190).  visible(wk2gt_1,190). orientation(wk2gt_2,141,190). coords(wk2gt_2,98,62,190).  visible(wk2gt_2,190).
orientation(wk2gt_1,129,191). coords(wk2gt_1,115,53,191).  visible(wk2gt_1,191). orientation(wk2gt_2,141,191). coords(wk2gt_2,98,62,191).  visible(wk2gt_2,191).
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
happensAt(disappear(wk2gt_1),204). happensAt(disappear(wk2gt_2),204).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=180..204.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=180..204.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),180). :- holdsAt(meeting(wk2gt_2,wk2gt_1),180).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),181). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),181).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),182). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),182).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),183). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),183).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),184). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),184).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),185). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),185).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),186). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),186).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),187). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),187).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),188). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),188).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),189). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),189).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),190). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),190).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),191). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),191).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),192). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),192).
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
:- holdsAt(meeting(wk2gt_1,wk2gt_2),205). :- holdsAt(meeting(wk2gt_2,wk2gt_1),205).
:- simpleEvent(T,P,walking), T>180 , T<204.
:- simpleEvent(T,P,running), T>180 , T<204.

#show simpleEvent/3.