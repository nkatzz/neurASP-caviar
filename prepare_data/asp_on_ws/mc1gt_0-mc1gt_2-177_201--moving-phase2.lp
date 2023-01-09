

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

	person(mc1gt_0;mc1gt_2).
orientation(mc1gt_0,147,177). coords(mc1gt_0,120,111,177).  visible(mc1gt_0,177). orientation(mc1gt_2,144,177). coords(mc1gt_2,88,113,177).  visible(mc1gt_2,177).
orientation(mc1gt_0,147,178). coords(mc1gt_0,121,111,178).  visible(mc1gt_0,178). orientation(mc1gt_2,144,178). coords(mc1gt_2,89,114,178).  visible(mc1gt_2,178).
orientation(mc1gt_0,147,179). coords(mc1gt_0,121,111,179).  visible(mc1gt_0,179). orientation(mc1gt_2,144,179). coords(mc1gt_2,90,114,179).  visible(mc1gt_2,179).
orientation(mc1gt_0,147,180). coords(mc1gt_0,122,111,180).  visible(mc1gt_0,180). orientation(mc1gt_2,144,180). coords(mc1gt_2,91,114,180).  visible(mc1gt_2,180).
orientation(mc1gt_0,147,181). coords(mc1gt_0,122,111,181).  visible(mc1gt_0,181). orientation(mc1gt_2,144,181). coords(mc1gt_2,92,114,181).  visible(mc1gt_2,181).
orientation(mc1gt_0,147,182). coords(mc1gt_0,122,111,182).  visible(mc1gt_0,182). orientation(mc1gt_2,144,182). coords(mc1gt_2,92,114,182).  visible(mc1gt_2,182).
orientation(mc1gt_0,147,183). coords(mc1gt_0,122,112,183).  visible(mc1gt_0,183). orientation(mc1gt_2,145,183). coords(mc1gt_2,92,115,183).  visible(mc1gt_2,183).
orientation(mc1gt_0,147,184). coords(mc1gt_0,123,112,184).  visible(mc1gt_0,184). orientation(mc1gt_2,146,184). coords(mc1gt_2,92,115,184).  visible(mc1gt_2,184).
orientation(mc1gt_0,147,185). coords(mc1gt_0,123,113,185).  visible(mc1gt_0,185). orientation(mc1gt_2,147,185). coords(mc1gt_2,92,115,185).  visible(mc1gt_2,185).
orientation(mc1gt_0,147,186). coords(mc1gt_0,123,114,186).  visible(mc1gt_0,186). orientation(mc1gt_2,147,186). coords(mc1gt_2,93,115,186).  visible(mc1gt_2,186).
orientation(mc1gt_0,147,187). coords(mc1gt_0,124,114,187).  visible(mc1gt_0,187). orientation(mc1gt_2,147,187). coords(mc1gt_2,93,115,187).  visible(mc1gt_2,187).
orientation(mc1gt_0,147,188). coords(mc1gt_0,126,115,188).  visible(mc1gt_0,188). orientation(mc1gt_2,147,188). coords(mc1gt_2,93,115,188).  visible(mc1gt_2,188).
orientation(mc1gt_0,147,189). coords(mc1gt_0,127,115,189).  visible(mc1gt_0,189). orientation(mc1gt_2,147,189). coords(mc1gt_2,95,115,189).  visible(mc1gt_2,189).
orientation(mc1gt_0,147,190). coords(mc1gt_0,128,116,190).  visible(mc1gt_0,190). orientation(mc1gt_2,148,190). coords(mc1gt_2,97,115,190).  visible(mc1gt_2,190).
orientation(mc1gt_0,147,191). coords(mc1gt_0,129,116,191).  visible(mc1gt_0,191). orientation(mc1gt_2,149,191). coords(mc1gt_2,98,116,191).  visible(mc1gt_2,191).
orientation(mc1gt_0,147,192). coords(mc1gt_0,130,116,192).  visible(mc1gt_0,192). orientation(mc1gt_2,149,192). coords(mc1gt_2,99,116,192).  visible(mc1gt_2,192).
orientation(mc1gt_0,147,193). coords(mc1gt_0,130,117,193).  visible(mc1gt_0,193). orientation(mc1gt_2,149,193). coords(mc1gt_2,100,116,193).  visible(mc1gt_2,193).
orientation(mc1gt_0,147,194). coords(mc1gt_0,130,117,194).  visible(mc1gt_0,194). orientation(mc1gt_2,149,194). coords(mc1gt_2,101,117,194).  visible(mc1gt_2,194).
orientation(mc1gt_0,147,195). coords(mc1gt_0,131,117,195).  visible(mc1gt_0,195). orientation(mc1gt_2,148,195). coords(mc1gt_2,101,117,195).  visible(mc1gt_2,195).
orientation(mc1gt_0,147,196). coords(mc1gt_0,131,117,196).  visible(mc1gt_0,196). orientation(mc1gt_2,148,196). coords(mc1gt_2,102,117,196).  visible(mc1gt_2,196).
orientation(mc1gt_0,147,197). coords(mc1gt_0,131,117,197).  visible(mc1gt_0,197). orientation(mc1gt_2,148,197). coords(mc1gt_2,102,117,197).  visible(mc1gt_2,197).
orientation(mc1gt_0,147,198). coords(mc1gt_0,131,117,198).  visible(mc1gt_0,198). orientation(mc1gt_2,147,198). coords(mc1gt_2,102,117,198).  visible(mc1gt_2,198).
orientation(mc1gt_0,147,199). coords(mc1gt_0,132,117,199).  visible(mc1gt_0,199). orientation(mc1gt_2,147,199). coords(mc1gt_2,102,117,199).  visible(mc1gt_2,199).
orientation(mc1gt_0,147,200). coords(mc1gt_0,134,117,200).  visible(mc1gt_0,200). orientation(mc1gt_2,147,200). coords(mc1gt_2,103,118,200).  visible(mc1gt_2,200).
orientation(mc1gt_0,147,201). coords(mc1gt_0,136,118,201).  visible(mc1gt_0,201). orientation(mc1gt_2,147,201). coords(mc1gt_2,103,118,201).  visible(mc1gt_2,201).
orientation(mc1gt_0,147,202). coords(mc1gt_0,136,118,202).  visible(mc1gt_0,202). orientation(mc1gt_2,147,202). coords(mc1gt_2,103,118,202).  visible(mc1gt_2,202).
happensAt(disappear(mc1gt_0),201). happensAt(disappear(mc1gt_2),201).
1{simpleEvent(T,mc1gt_0,active); simpleEvent(T,mc1gt_0,inactive); simpleEvent(T,mc1gt_0,walking); simpleEvent(T,mc1gt_0,running)}1 :- T=177..201.
1{simpleEvent(T,mc1gt_2,active); simpleEvent(T,mc1gt_2,inactive); simpleEvent(T,mc1gt_2,walking); simpleEvent(T,mc1gt_2,running)}1 :- T=177..201.
:- holdsAt(moving(mc1gt_0,mc1gt_2),177). :- holdsAt(moving(mc1gt_2,mc1gt_0),177).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),178). :- not holdsAt(moving(mc1gt_2,mc1gt_0),178).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),179). :- not holdsAt(moving(mc1gt_2,mc1gt_0),179).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),180). :- not holdsAt(moving(mc1gt_2,mc1gt_0),180).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),181). :- not holdsAt(moving(mc1gt_2,mc1gt_0),181).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),182). :- not holdsAt(moving(mc1gt_2,mc1gt_0),182).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),183). :- not holdsAt(moving(mc1gt_2,mc1gt_0),183).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),184). :- not holdsAt(moving(mc1gt_2,mc1gt_0),184).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),185). :- not holdsAt(moving(mc1gt_2,mc1gt_0),185).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),186). :- not holdsAt(moving(mc1gt_2,mc1gt_0),186).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),187). :- not holdsAt(moving(mc1gt_2,mc1gt_0),187).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),188). :- not holdsAt(moving(mc1gt_2,mc1gt_0),188).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),189). :- not holdsAt(moving(mc1gt_2,mc1gt_0),189).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),190). :- not holdsAt(moving(mc1gt_2,mc1gt_0),190).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),191). :- not holdsAt(moving(mc1gt_2,mc1gt_0),191).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),192). :- not holdsAt(moving(mc1gt_2,mc1gt_0),192).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),193). :- not holdsAt(moving(mc1gt_2,mc1gt_0),193).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),194). :- not holdsAt(moving(mc1gt_2,mc1gt_0),194).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),195). :- not holdsAt(moving(mc1gt_2,mc1gt_0),195).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),196). :- not holdsAt(moving(mc1gt_2,mc1gt_0),196).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),197). :- not holdsAt(moving(mc1gt_2,mc1gt_0),197).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),198). :- not holdsAt(moving(mc1gt_2,mc1gt_0),198).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),199). :- not holdsAt(moving(mc1gt_2,mc1gt_0),199).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),200). :- not holdsAt(moving(mc1gt_2,mc1gt_0),200).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),201). :- not holdsAt(moving(mc1gt_2,mc1gt_0),201).
:- holdsAt(moving(mc1gt_0,mc1gt_2),202). :- holdsAt(moving(mc1gt_2,mc1gt_0),202).
:- simpleEvent(T,P,inactive), T>177 , T<201.
:- simpleEvent(T,P,running), T>177 , T<201.

#show simpleEvent/3.