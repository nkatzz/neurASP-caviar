

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
orientation(mws1gt_1,155,176). coords(mws1gt_1,150,138,176).  visible(mws1gt_1,176). orientation(mws1gt_3,154,176). coords(mws1gt_3,157,154,176).  visible(mws1gt_3,176).
orientation(mws1gt_1,155,177). coords(mws1gt_1,151,137,177).  visible(mws1gt_1,177). orientation(mws1gt_3,154,177). coords(mws1gt_3,157,154,177).  visible(mws1gt_3,177).
orientation(mws1gt_1,155,178). coords(mws1gt_1,151,136,178).  visible(mws1gt_1,178). orientation(mws1gt_3,154,178). coords(mws1gt_3,156,154,178).  visible(mws1gt_3,178).
orientation(mws1gt_1,155,179). coords(mws1gt_1,151,136,179).  visible(mws1gt_1,179). orientation(mws1gt_3,154,179). coords(mws1gt_3,156,154,179).  visible(mws1gt_3,179).
orientation(mws1gt_1,155,180). coords(mws1gt_1,151,136,180).  visible(mws1gt_1,180). orientation(mws1gt_3,154,180). coords(mws1gt_3,156,153,180).  visible(mws1gt_3,180).
orientation(mws1gt_1,155,181). coords(mws1gt_1,151,136,181).  visible(mws1gt_1,181). orientation(mws1gt_3,154,181). coords(mws1gt_3,156,153,181).  visible(mws1gt_3,181).
orientation(mws1gt_1,155,182). coords(mws1gt_1,150,135,182).  visible(mws1gt_1,182). orientation(mws1gt_3,154,182). coords(mws1gt_3,156,153,182).  visible(mws1gt_3,182).
orientation(mws1gt_1,155,183). coords(mws1gt_1,150,135,183).  visible(mws1gt_1,183). orientation(mws1gt_3,154,183). coords(mws1gt_3,155,153,183).  visible(mws1gt_3,183).
orientation(mws1gt_1,155,184). coords(mws1gt_1,149,136,184).  visible(mws1gt_1,184). orientation(mws1gt_3,154,184). coords(mws1gt_3,155,153,184).  visible(mws1gt_3,184).
orientation(mws1gt_1,155,185). coords(mws1gt_1,149,136,185).  visible(mws1gt_1,185). orientation(mws1gt_3,154,185). coords(mws1gt_3,154,153,185).  visible(mws1gt_3,185).
orientation(mws1gt_1,155,186). coords(mws1gt_1,148,136,186).  visible(mws1gt_1,186). orientation(mws1gt_3,154,186). coords(mws1gt_3,154,153,186).  visible(mws1gt_3,186).
orientation(mws1gt_1,155,187). coords(mws1gt_1,148,136,187).  visible(mws1gt_1,187). orientation(mws1gt_3,154,187). coords(mws1gt_3,154,153,187).  visible(mws1gt_3,187).
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
orientation(mws1gt_1,155,201). coords(mws1gt_1,140,133,201).  visible(mws1gt_1,201). orientation(mws1gt_3,162,201). coords(mws1gt_3,140,154,201).  visible(mws1gt_3,201).
happensAt(disappear(mws1gt_1),200). happensAt(disappear(mws1gt_3),200).
1{simpleEvent(T,mws1gt_1,active); simpleEvent(T,mws1gt_1,inactive); simpleEvent(T,mws1gt_1,walking); simpleEvent(T,mws1gt_1,running)}1 :- T=176..200.
1{simpleEvent(T,mws1gt_3,active); simpleEvent(T,mws1gt_3,inactive); simpleEvent(T,mws1gt_3,walking); simpleEvent(T,mws1gt_3,running)}1 :- T=176..200.
:- holdsAt(moving(mws1gt_1,mws1gt_3),176). :- holdsAt(moving(mws1gt_3,mws1gt_1),176).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),177). :- not holdsAt(moving(mws1gt_3,mws1gt_1),177).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),178). :- not holdsAt(moving(mws1gt_3,mws1gt_1),178).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),179). :- not holdsAt(moving(mws1gt_3,mws1gt_1),179).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),180). :- not holdsAt(moving(mws1gt_3,mws1gt_1),180).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),181). :- not holdsAt(moving(mws1gt_3,mws1gt_1),181).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),182). :- not holdsAt(moving(mws1gt_3,mws1gt_1),182).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),183). :- not holdsAt(moving(mws1gt_3,mws1gt_1),183).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),184). :- not holdsAt(moving(mws1gt_3,mws1gt_1),184).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),185). :- not holdsAt(moving(mws1gt_3,mws1gt_1),185).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),186). :- not holdsAt(moving(mws1gt_3,mws1gt_1),186).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),187). :- not holdsAt(moving(mws1gt_3,mws1gt_1),187).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),188). :- not holdsAt(moving(mws1gt_3,mws1gt_1),188).
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
:- holdsAt(moving(mws1gt_1,mws1gt_3),201). :- holdsAt(moving(mws1gt_3,mws1gt_1),201).
:- simpleEvent(T,P,inactive), T>176 , T<200.
:- simpleEvent(T,P,running), T>176 , T<200.

#show simpleEvent/3.