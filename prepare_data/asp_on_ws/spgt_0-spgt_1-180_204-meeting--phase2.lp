

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
orientation(spgt_0,148,180). coords(spgt_0,113,114,180).  visible(spgt_0,180). orientation(spgt_1,148,180). coords(spgt_1,108,103,180).  visible(spgt_1,180).
orientation(spgt_0,148,181). coords(spgt_0,113,114,181).  visible(spgt_0,181). orientation(spgt_1,148,181). coords(spgt_1,108,103,181).  visible(spgt_1,181).
orientation(spgt_0,148,182). coords(spgt_0,113,114,182).  visible(spgt_0,182). orientation(spgt_1,148,182). coords(spgt_1,108,103,182).  visible(spgt_1,182).
orientation(spgt_0,148,183). coords(spgt_0,113,114,183).  visible(spgt_0,183). orientation(spgt_1,148,183). coords(spgt_1,108,103,183).  visible(spgt_1,183).
orientation(spgt_0,148,184). coords(spgt_0,113,114,184).  visible(spgt_0,184). orientation(spgt_1,148,184). coords(spgt_1,108,103,184).  visible(spgt_1,184).
orientation(spgt_0,148,185). coords(spgt_0,113,114,185).  visible(spgt_0,185). orientation(spgt_1,148,185). coords(spgt_1,108,103,185).  visible(spgt_1,185).
orientation(spgt_0,148,186). coords(spgt_0,113,114,186).  visible(spgt_0,186). orientation(spgt_1,148,186). coords(spgt_1,108,103,186).  visible(spgt_1,186).
orientation(spgt_0,148,187). coords(spgt_0,113,114,187).  visible(spgt_0,187). orientation(spgt_1,148,187). coords(spgt_1,108,103,187).  visible(spgt_1,187).
orientation(spgt_0,148,188). coords(spgt_0,113,114,188).  visible(spgt_0,188). orientation(spgt_1,148,188). coords(spgt_1,108,103,188).  visible(spgt_1,188).
orientation(spgt_0,148,189). coords(spgt_0,113,114,189).  visible(spgt_0,189). orientation(spgt_1,148,189). coords(spgt_1,108,103,189).  visible(spgt_1,189).
orientation(spgt_0,148,190). coords(spgt_0,113,114,190).  visible(spgt_0,190). orientation(spgt_1,148,190). coords(spgt_1,108,103,190).  visible(spgt_1,190).
orientation(spgt_0,148,191). coords(spgt_0,113,114,191).  visible(spgt_0,191). orientation(spgt_1,148,191). coords(spgt_1,108,103,191).  visible(spgt_1,191).
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
happensAt(disappear(spgt_0),204). happensAt(disappear(spgt_1),204).
1{simpleEvent(T,spgt_0,active); simpleEvent(T,spgt_0,inactive); simpleEvent(T,spgt_0,walking); simpleEvent(T,spgt_0,running)}1 :- T=180..204.
1{simpleEvent(T,spgt_1,active); simpleEvent(T,spgt_1,inactive); simpleEvent(T,spgt_1,walking); simpleEvent(T,spgt_1,running)}1 :- T=180..204.
:- holdsAt(meeting(spgt_0,spgt_1),180). :- holdsAt(meeting(spgt_1,spgt_0),180).
:- not holdsAt(meeting(spgt_0,spgt_1),181). :- not holdsAt(meeting(spgt_1,spgt_0),181).
:- not holdsAt(meeting(spgt_0,spgt_1),182). :- not holdsAt(meeting(spgt_1,spgt_0),182).
:- not holdsAt(meeting(spgt_0,spgt_1),183). :- not holdsAt(meeting(spgt_1,spgt_0),183).
:- not holdsAt(meeting(spgt_0,spgt_1),184). :- not holdsAt(meeting(spgt_1,spgt_0),184).
:- not holdsAt(meeting(spgt_0,spgt_1),185). :- not holdsAt(meeting(spgt_1,spgt_0),185).
:- not holdsAt(meeting(spgt_0,spgt_1),186). :- not holdsAt(meeting(spgt_1,spgt_0),186).
:- not holdsAt(meeting(spgt_0,spgt_1),187). :- not holdsAt(meeting(spgt_1,spgt_0),187).
:- not holdsAt(meeting(spgt_0,spgt_1),188). :- not holdsAt(meeting(spgt_1,spgt_0),188).
:- not holdsAt(meeting(spgt_0,spgt_1),189). :- not holdsAt(meeting(spgt_1,spgt_0),189).
:- not holdsAt(meeting(spgt_0,spgt_1),190). :- not holdsAt(meeting(spgt_1,spgt_0),190).
:- not holdsAt(meeting(spgt_0,spgt_1),191). :- not holdsAt(meeting(spgt_1,spgt_0),191).
:- not holdsAt(meeting(spgt_0,spgt_1),192). :- not holdsAt(meeting(spgt_1,spgt_0),192).
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
:- holdsAt(meeting(spgt_0,spgt_1),205). :- holdsAt(meeting(spgt_1,spgt_0),205).
:- simpleEvent(T,P,walking), T>181 , T<181.
:- simpleEvent(T,P,running), T>181 , T<181.

#show simpleEvent/3.