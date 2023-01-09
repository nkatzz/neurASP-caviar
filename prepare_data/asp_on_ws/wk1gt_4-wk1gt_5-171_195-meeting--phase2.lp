

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
orientation(wk1gt_4,158,171). coords(wk1gt_4,47,181,171).  visible(wk1gt_4,171). orientation(wk1gt_5,162,171). coords(wk1gt_5,48,166,171).  visible(wk1gt_5,171).
orientation(wk1gt_4,165,172). coords(wk1gt_4,47,181,172).  visible(wk1gt_4,172). orientation(wk1gt_5,162,172). coords(wk1gt_5,48,166,172).  visible(wk1gt_5,172).
orientation(wk1gt_4,165,173). coords(wk1gt_4,47,181,173).  visible(wk1gt_4,173). orientation(wk1gt_5,162,173). coords(wk1gt_5,48,166,173).  visible(wk1gt_5,173).
orientation(wk1gt_4,163,174). coords(wk1gt_4,47,181,174).  visible(wk1gt_4,174). orientation(wk1gt_5,162,174). coords(wk1gt_5,48,166,174).  visible(wk1gt_5,174).
orientation(wk1gt_4,162,175). coords(wk1gt_4,47,181,175).  visible(wk1gt_4,175). orientation(wk1gt_5,162,175). coords(wk1gt_5,49,166,175).  visible(wk1gt_5,175).
orientation(wk1gt_4,161,176). coords(wk1gt_4,47,181,176).  visible(wk1gt_4,176). orientation(wk1gt_5,162,176). coords(wk1gt_5,49,166,176).  visible(wk1gt_5,176).
orientation(wk1gt_4,161,177). coords(wk1gt_4,47,181,177).  visible(wk1gt_4,177). orientation(wk1gt_5,162,177). coords(wk1gt_5,49,166,177).  visible(wk1gt_5,177).
orientation(wk1gt_4,163,178). coords(wk1gt_4,47,181,178).  visible(wk1gt_4,178). orientation(wk1gt_5,162,178). coords(wk1gt_5,49,166,178).  visible(wk1gt_5,178).
orientation(wk1gt_4,164,179). coords(wk1gt_4,47,181,179).  visible(wk1gt_4,179). orientation(wk1gt_5,162,179). coords(wk1gt_5,49,166,179).  visible(wk1gt_5,179).
orientation(wk1gt_4,160,180). coords(wk1gt_4,47,181,180).  visible(wk1gt_4,180). orientation(wk1gt_5,162,180). coords(wk1gt_5,49,166,180).  visible(wk1gt_5,180).
orientation(wk1gt_4,158,181). coords(wk1gt_4,47,181,181).  visible(wk1gt_4,181). orientation(wk1gt_5,162,181). coords(wk1gt_5,49,166,181).  visible(wk1gt_5,181).
orientation(wk1gt_4,163,182). coords(wk1gt_4,47,181,182).  visible(wk1gt_4,182). orientation(wk1gt_5,162,182). coords(wk1gt_5,49,166,182).  visible(wk1gt_5,182).
orientation(wk1gt_4,162,183). coords(wk1gt_4,47,181,183).  visible(wk1gt_4,183). orientation(wk1gt_5,162,183). coords(wk1gt_5,49,166,183).  visible(wk1gt_5,183).
orientation(wk1gt_4,161,184). coords(wk1gt_4,47,181,184).  visible(wk1gt_4,184). orientation(wk1gt_5,162,184). coords(wk1gt_5,49,166,184).  visible(wk1gt_5,184).
orientation(wk1gt_4,164,185). coords(wk1gt_4,47,181,185).  visible(wk1gt_4,185). orientation(wk1gt_5,162,185). coords(wk1gt_5,49,166,185).  visible(wk1gt_5,185).
orientation(wk1gt_4,164,186). coords(wk1gt_4,47,181,186).  visible(wk1gt_4,186). orientation(wk1gt_5,162,186). coords(wk1gt_5,49,166,186).  visible(wk1gt_5,186).
orientation(wk1gt_4,159,187). coords(wk1gt_4,47,181,187).  visible(wk1gt_4,187). orientation(wk1gt_5,162,187). coords(wk1gt_5,49,166,187).  visible(wk1gt_5,187).
orientation(wk1gt_4,164,188). coords(wk1gt_4,47,181,188).  visible(wk1gt_4,188). orientation(wk1gt_5,162,188). coords(wk1gt_5,49,166,188).  visible(wk1gt_5,188).
orientation(wk1gt_4,164,189). coords(wk1gt_4,47,181,189).  visible(wk1gt_4,189). orientation(wk1gt_5,162,189). coords(wk1gt_5,49,166,189).  visible(wk1gt_5,189).
orientation(wk1gt_4,159,190). coords(wk1gt_4,47,181,190).  visible(wk1gt_4,190). orientation(wk1gt_5,162,190). coords(wk1gt_5,49,166,190).  visible(wk1gt_5,190).
orientation(wk1gt_4,167,191). coords(wk1gt_4,47,181,191).  visible(wk1gt_4,191). orientation(wk1gt_5,162,191). coords(wk1gt_5,49,166,191).  visible(wk1gt_5,191).
orientation(wk1gt_4,168,192). coords(wk1gt_4,47,181,192).  visible(wk1gt_4,192). orientation(wk1gt_5,162,192). coords(wk1gt_5,49,166,192).  visible(wk1gt_5,192).
orientation(wk1gt_4,168,193). coords(wk1gt_4,47,181,193).  visible(wk1gt_4,193). orientation(wk1gt_5,162,193). coords(wk1gt_5,49,166,193).  visible(wk1gt_5,193).
orientation(wk1gt_4,167,194). coords(wk1gt_4,47,181,194).  visible(wk1gt_4,194). orientation(wk1gt_5,162,194). coords(wk1gt_5,49,166,194).  visible(wk1gt_5,194).
orientation(wk1gt_4,166,195). coords(wk1gt_4,47,181,195).  visible(wk1gt_4,195). orientation(wk1gt_5,162,195). coords(wk1gt_5,49,166,195).  visible(wk1gt_5,195).
orientation(wk1gt_4,166,196). coords(wk1gt_4,47,181,196).  visible(wk1gt_4,196). orientation(wk1gt_5,162,196). coords(wk1gt_5,49,166,196).  visible(wk1gt_5,196).
happensAt(disappear(wk1gt_4),195). happensAt(disappear(wk1gt_5),195).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=171..195.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=171..195.
:- holdsAt(meeting(wk1gt_4,wk1gt_5),171). :- holdsAt(meeting(wk1gt_5,wk1gt_4),171).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),172). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),172).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),173). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),173).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),174). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),174).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),175). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),175).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),176). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),176).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),177). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),177).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),178). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),178).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),179). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),179).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),180). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),180).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),181). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),181).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),182). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),182).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),183). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),183).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),184). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),184).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),185). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),185).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),186). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),186).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),187). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),187).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),188). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),188).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),189). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),189).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),190). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),190).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),191). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),191).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),192). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),192).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),193). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),193).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),194). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),194).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),195). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),195).
:- holdsAt(meeting(wk1gt_4,wk1gt_5),196). :- holdsAt(meeting(wk1gt_5,wk1gt_4),196).
:- simpleEvent(T,P,walking), T>184 , T<180.
:- simpleEvent(T,P,running), T>184 , T<180.

#show simpleEvent/3.