
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


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(fcgt_3,159,180). coords(fcgt_3,103,163,180).  visible(fcgt_3,180). happensAt(inactive(fcgt_3),180). orientation(fcgt_4,163,180). coords(fcgt_4,93,205,180).  visible(fcgt_4,180). happensAt(walking(fcgt_4),180).
orientation(fcgt_3,159,181). coords(fcgt_3,103,163,181).  visible(fcgt_3,181). happensAt(inactive(fcgt_3),181). orientation(fcgt_4,163,181). coords(fcgt_4,92,204,181).  visible(fcgt_4,181). happensAt(walking(fcgt_4),181).
orientation(fcgt_3,159,182). coords(fcgt_3,103,163,182).  visible(fcgt_3,182). happensAt(inactive(fcgt_3),182). orientation(fcgt_4,169,182). coords(fcgt_4,92,204,182).  visible(fcgt_4,182). happensAt(walking(fcgt_4),182).
orientation(fcgt_3,159,183). coords(fcgt_3,103,163,183).  visible(fcgt_3,183). happensAt(inactive(fcgt_3),183). orientation(fcgt_4,169,183). coords(fcgt_4,92,204,183).  visible(fcgt_4,183). happensAt(walking(fcgt_4),183).
orientation(fcgt_3,159,184). coords(fcgt_3,103,163,184).  visible(fcgt_3,184). happensAt(inactive(fcgt_3),184). orientation(fcgt_4,169,184). coords(fcgt_4,92,204,184).  visible(fcgt_4,184). happensAt(walking(fcgt_4),184).
orientation(fcgt_3,159,185). coords(fcgt_3,103,163,185).  visible(fcgt_3,185). happensAt(inactive(fcgt_3),185). orientation(fcgt_4,174,185). coords(fcgt_4,92,204,185).  visible(fcgt_4,185). happensAt(walking(fcgt_4),185).
orientation(fcgt_3,159,186). coords(fcgt_3,103,163,186).  visible(fcgt_3,186). happensAt(inactive(fcgt_3),186). orientation(fcgt_4,174,186). coords(fcgt_4,92,204,186).  visible(fcgt_4,186). happensAt(walking(fcgt_4),186).
orientation(fcgt_3,159,187). coords(fcgt_3,103,163,187).  visible(fcgt_3,187). happensAt(inactive(fcgt_3),187). orientation(fcgt_4,174,187). coords(fcgt_4,92,204,187).  visible(fcgt_4,187). happensAt(walking(fcgt_4),187).
orientation(fcgt_3,159,188). coords(fcgt_3,103,163,188).  visible(fcgt_3,188). happensAt(inactive(fcgt_3),188). orientation(fcgt_4,174,188). coords(fcgt_4,91,204,188).  visible(fcgt_4,188). happensAt(walking(fcgt_4),188).
orientation(fcgt_3,159,189). coords(fcgt_3,103,163,189).  visible(fcgt_3,189). happensAt(inactive(fcgt_3),189). orientation(fcgt_4,174,189). coords(fcgt_4,91,204,189).  visible(fcgt_4,189). happensAt(walking(fcgt_4),189).
orientation(fcgt_3,159,190). coords(fcgt_3,103,163,190).  visible(fcgt_3,190). happensAt(inactive(fcgt_3),190). orientation(fcgt_4,174,190). coords(fcgt_4,91,204,190).  visible(fcgt_4,190). happensAt(walking(fcgt_4),190).
orientation(fcgt_3,159,191). coords(fcgt_3,103,163,191).  visible(fcgt_3,191). happensAt(inactive(fcgt_3),191). orientation(fcgt_4,174,191). coords(fcgt_4,91,204,191).  visible(fcgt_4,191). happensAt(walking(fcgt_4),191).
orientation(fcgt_3,159,192). coords(fcgt_3,103,163,192).  visible(fcgt_3,192). happensAt(inactive(fcgt_3),192). orientation(fcgt_4,174,192). coords(fcgt_4,90,204,192).  visible(fcgt_4,192). happensAt(walking(fcgt_4),192).
orientation(fcgt_3,159,193). coords(fcgt_3,103,163,193).  visible(fcgt_3,193). happensAt(inactive(fcgt_3),193). orientation(fcgt_4,174,193). coords(fcgt_4,89,204,193).  visible(fcgt_4,193). happensAt(walking(fcgt_4),193).
orientation(fcgt_3,159,194). coords(fcgt_3,103,163,194).  visible(fcgt_3,194). happensAt(inactive(fcgt_3),194). orientation(fcgt_4,174,194). coords(fcgt_4,88,204,194).  visible(fcgt_4,194). happensAt(walking(fcgt_4),194).
orientation(fcgt_3,159,195). coords(fcgt_3,103,163,195).  visible(fcgt_3,195). happensAt(inactive(fcgt_3),195). orientation(fcgt_4,174,195). coords(fcgt_4,87,204,195).  visible(fcgt_4,195). happensAt(walking(fcgt_4),195).
orientation(fcgt_3,159,196). coords(fcgt_3,103,163,196).  visible(fcgt_3,196). happensAt(inactive(fcgt_3),196). orientation(fcgt_4,174,196). coords(fcgt_4,87,204,196).  visible(fcgt_4,196). happensAt(walking(fcgt_4),196).
orientation(fcgt_3,159,197). coords(fcgt_3,103,163,197).  visible(fcgt_3,197). happensAt(inactive(fcgt_3),197). orientation(fcgt_4,178,197). coords(fcgt_4,86,204,197).  visible(fcgt_4,197). happensAt(walking(fcgt_4),197).
orientation(fcgt_3,159,198). coords(fcgt_3,103,163,198).  visible(fcgt_3,198). happensAt(inactive(fcgt_3),198). orientation(fcgt_4,179,198). coords(fcgt_4,85,204,198).  visible(fcgt_4,198). happensAt(walking(fcgt_4),198).
orientation(fcgt_3,159,199). coords(fcgt_3,103,163,199).  visible(fcgt_3,199). happensAt(inactive(fcgt_3),199). orientation(fcgt_4,1,199). coords(fcgt_4,85,204,199).  visible(fcgt_4,199). happensAt(walking(fcgt_4),199).
orientation(fcgt_3,159,200). coords(fcgt_3,103,163,200).  visible(fcgt_3,200). happensAt(inactive(fcgt_3),200). orientation(fcgt_4,1,200). coords(fcgt_4,85,204,200).  visible(fcgt_4,200). happensAt(walking(fcgt_4),200).
orientation(fcgt_3,159,201). coords(fcgt_3,102,163,201).  visible(fcgt_3,201). happensAt(inactive(fcgt_3),201). orientation(fcgt_4,1,201). coords(fcgt_4,85,204,201).  visible(fcgt_4,201). happensAt(walking(fcgt_4),201).
orientation(fcgt_3,159,202). coords(fcgt_3,102,163,202).  visible(fcgt_3,202). happensAt(walking(fcgt_3),202). orientation(fcgt_4,0,202). coords(fcgt_4,85,204,202).  visible(fcgt_4,202). happensAt(walking(fcgt_4),202).
orientation(fcgt_3,159,203). coords(fcgt_3,101,163,203).  visible(fcgt_3,203). happensAt(walking(fcgt_3),203). orientation(fcgt_4,0,203). coords(fcgt_4,85,205,203).  visible(fcgt_4,203). happensAt(walking(fcgt_4),203).
orientation(fcgt_3,159,204). coords(fcgt_3,101,164,204).  visible(fcgt_3,204). happensAt(walking(fcgt_3),204). orientation(fcgt_4,0,204). coords(fcgt_4,85,205,204).  visible(fcgt_4,204). happensAt(walking(fcgt_4),204).
#show holdsAt/2.