

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

	person(mc1gt_0;mc1gt_1).
orientation(mc1gt_0,147,172). coords(mc1gt_0,113,110,172).  visible(mc1gt_0,172). orientation(mc1gt_1,145,172). coords(mc1gt_1,138,90,172).  visible(mc1gt_1,172).
orientation(mc1gt_0,147,173). coords(mc1gt_0,114,110,173).  visible(mc1gt_0,173). orientation(mc1gt_1,145,173). coords(mc1gt_1,138,90,173).  visible(mc1gt_1,173).
orientation(mc1gt_0,147,174). coords(mc1gt_0,116,110,174).  visible(mc1gt_0,174). orientation(mc1gt_1,145,174). coords(mc1gt_1,138,90,174).  visible(mc1gt_1,174).
orientation(mc1gt_0,147,175). coords(mc1gt_0,118,111,175).  visible(mc1gt_0,175). orientation(mc1gt_1,145,175). coords(mc1gt_1,139,90,175).  visible(mc1gt_1,175).
orientation(mc1gt_0,147,176). coords(mc1gt_0,119,111,176).  visible(mc1gt_0,176). orientation(mc1gt_1,145,176). coords(mc1gt_1,139,90,176).  visible(mc1gt_1,176).
orientation(mc1gt_0,147,177). coords(mc1gt_0,120,111,177).  visible(mc1gt_0,177). orientation(mc1gt_1,145,177). coords(mc1gt_1,139,91,177).  visible(mc1gt_1,177).
orientation(mc1gt_0,147,178). coords(mc1gt_0,121,111,178).  visible(mc1gt_0,178). orientation(mc1gt_1,145,178). coords(mc1gt_1,139,91,178).  visible(mc1gt_1,178).
orientation(mc1gt_0,147,179). coords(mc1gt_0,121,111,179).  visible(mc1gt_0,179). orientation(mc1gt_1,145,179). coords(mc1gt_1,140,91,179).  visible(mc1gt_1,179).
orientation(mc1gt_0,147,180). coords(mc1gt_0,122,111,180).  visible(mc1gt_0,180). orientation(mc1gt_1,145,180). coords(mc1gt_1,140,91,180).  visible(mc1gt_1,180).
orientation(mc1gt_0,147,181). coords(mc1gt_0,122,111,181).  visible(mc1gt_0,181). orientation(mc1gt_1,145,181). coords(mc1gt_1,140,92,181).  visible(mc1gt_1,181).
orientation(mc1gt_0,147,182). coords(mc1gt_0,122,111,182).  visible(mc1gt_0,182). orientation(mc1gt_1,145,182). coords(mc1gt_1,142,92,182).  visible(mc1gt_1,182).
orientation(mc1gt_0,147,183). coords(mc1gt_0,122,112,183).  visible(mc1gt_0,183). orientation(mc1gt_1,145,183). coords(mc1gt_1,144,92,183).  visible(mc1gt_1,183).
orientation(mc1gt_0,147,184). coords(mc1gt_0,123,112,184).  visible(mc1gt_0,184). orientation(mc1gt_1,145,184). coords(mc1gt_1,146,92,184).  visible(mc1gt_1,184).
orientation(mc1gt_0,147,185). coords(mc1gt_0,123,113,185).  visible(mc1gt_0,185). orientation(mc1gt_1,145,185). coords(mc1gt_1,147,94,185).  visible(mc1gt_1,185).
orientation(mc1gt_0,147,186). coords(mc1gt_0,123,114,186).  visible(mc1gt_0,186). orientation(mc1gt_1,145,186). coords(mc1gt_1,148,94,186).  visible(mc1gt_1,186).
orientation(mc1gt_0,147,187). coords(mc1gt_0,124,114,187).  visible(mc1gt_0,187). orientation(mc1gt_1,145,187). coords(mc1gt_1,149,95,187).  visible(mc1gt_1,187).
orientation(mc1gt_0,147,188). coords(mc1gt_0,126,115,188).  visible(mc1gt_0,188). orientation(mc1gt_1,143,188). coords(mc1gt_1,150,96,188).  visible(mc1gt_1,188).
orientation(mc1gt_0,147,189). coords(mc1gt_0,127,115,189).  visible(mc1gt_0,189). orientation(mc1gt_1,141,189). coords(mc1gt_1,151,96,189).  visible(mc1gt_1,189).
orientation(mc1gt_0,147,190). coords(mc1gt_0,128,116,190).  visible(mc1gt_0,190). orientation(mc1gt_1,139,190). coords(mc1gt_1,151,97,190).  visible(mc1gt_1,190).
orientation(mc1gt_0,147,191). coords(mc1gt_0,129,116,191).  visible(mc1gt_0,191). orientation(mc1gt_1,139,191). coords(mc1gt_1,152,97,191).  visible(mc1gt_1,191).
orientation(mc1gt_0,147,192). coords(mc1gt_0,130,116,192).  visible(mc1gt_0,192). orientation(mc1gt_1,140,192). coords(mc1gt_1,152,97,192).  visible(mc1gt_1,192).
orientation(mc1gt_0,147,193). coords(mc1gt_0,130,117,193).  visible(mc1gt_0,193). orientation(mc1gt_1,140,193). coords(mc1gt_1,152,97,193).  visible(mc1gt_1,193).
orientation(mc1gt_0,147,194). coords(mc1gt_0,130,117,194).  visible(mc1gt_0,194). orientation(mc1gt_1,140,194). coords(mc1gt_1,152,97,194).  visible(mc1gt_1,194).
orientation(mc1gt_0,147,195). coords(mc1gt_0,131,117,195).  visible(mc1gt_0,195). orientation(mc1gt_1,140,195). coords(mc1gt_1,152,97,195).  visible(mc1gt_1,195).
orientation(mc1gt_0,147,196). coords(mc1gt_0,131,117,196).  visible(mc1gt_0,196). orientation(mc1gt_1,140,196). coords(mc1gt_1,153,98,196).  visible(mc1gt_1,196).
orientation(mc1gt_0,147,197). coords(mc1gt_0,131,117,197).  visible(mc1gt_0,197). orientation(mc1gt_1,140,197). coords(mc1gt_1,153,98,197).  visible(mc1gt_1,197).
happensAt(disappear(mc1gt_0),196). happensAt(disappear(mc1gt_1),196).
1{simpleEvent(T,mc1gt_0,active); simpleEvent(T,mc1gt_0,inactive); simpleEvent(T,mc1gt_0,walking); simpleEvent(T,mc1gt_0,running)}1 :- T=172..196.
1{simpleEvent(T,mc1gt_1,active); simpleEvent(T,mc1gt_1,inactive); simpleEvent(T,mc1gt_1,walking); simpleEvent(T,mc1gt_1,running)}1 :- T=172..196.
:- holdsAt(moving(mc1gt_0,mc1gt_1),172). :- holdsAt(moving(mc1gt_1,mc1gt_0),172).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),173). :- not holdsAt(moving(mc1gt_1,mc1gt_0),173).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),174). :- not holdsAt(moving(mc1gt_1,mc1gt_0),174).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),175). :- not holdsAt(moving(mc1gt_1,mc1gt_0),175).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),176). :- not holdsAt(moving(mc1gt_1,mc1gt_0),176).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),177). :- not holdsAt(moving(mc1gt_1,mc1gt_0),177).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),178). :- not holdsAt(moving(mc1gt_1,mc1gt_0),178).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),179). :- not holdsAt(moving(mc1gt_1,mc1gt_0),179).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),180). :- not holdsAt(moving(mc1gt_1,mc1gt_0),180).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),181). :- not holdsAt(moving(mc1gt_1,mc1gt_0),181).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),182). :- not holdsAt(moving(mc1gt_1,mc1gt_0),182).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),183). :- not holdsAt(moving(mc1gt_1,mc1gt_0),183).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),184). :- not holdsAt(moving(mc1gt_1,mc1gt_0),184).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),185). :- not holdsAt(moving(mc1gt_1,mc1gt_0),185).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),186). :- not holdsAt(moving(mc1gt_1,mc1gt_0),186).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),187). :- not holdsAt(moving(mc1gt_1,mc1gt_0),187).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),188). :- not holdsAt(moving(mc1gt_1,mc1gt_0),188).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),189). :- not holdsAt(moving(mc1gt_1,mc1gt_0),189).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),190). :- not holdsAt(moving(mc1gt_1,mc1gt_0),190).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),191). :- not holdsAt(moving(mc1gt_1,mc1gt_0),191).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),192). :- not holdsAt(moving(mc1gt_1,mc1gt_0),192).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),193). :- not holdsAt(moving(mc1gt_1,mc1gt_0),193).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),194). :- not holdsAt(moving(mc1gt_1,mc1gt_0),194).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),195). :- not holdsAt(moving(mc1gt_1,mc1gt_0),195).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),196). :- not holdsAt(moving(mc1gt_1,mc1gt_0),196).
:- holdsAt(moving(mc1gt_0,mc1gt_1),197). :- holdsAt(moving(mc1gt_1,mc1gt_0),197).
:- simpleEvent(T,P,inactive), T>172 , T<196.
:- simpleEvent(T,P,running), T>172 , T<196.

#show simpleEvent/3.