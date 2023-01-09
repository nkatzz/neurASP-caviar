
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

	orientation(mc1gt_0,147,180). coords(mc1gt_0,122,111,180).  visible(mc1gt_0,180). happensAt(walking(mc1gt_0),180). orientation(mc1gt_3,146,180). coords(mc1gt_3,113,81,180).  visible(mc1gt_3,180). happensAt(walking(mc1gt_3),180).
orientation(mc1gt_0,147,181). coords(mc1gt_0,122,111,181).  visible(mc1gt_0,181). happensAt(walking(mc1gt_0),181). orientation(mc1gt_3,146,181). coords(mc1gt_3,113,82,181).  visible(mc1gt_3,181). happensAt(walking(mc1gt_3),181).
orientation(mc1gt_0,147,182). coords(mc1gt_0,122,111,182).  visible(mc1gt_0,182). happensAt(walking(mc1gt_0),182). orientation(mc1gt_3,146,182). coords(mc1gt_3,114,82,182).  visible(mc1gt_3,182). happensAt(walking(mc1gt_3),182).
orientation(mc1gt_0,147,183). coords(mc1gt_0,122,112,183).  visible(mc1gt_0,183). happensAt(walking(mc1gt_0),183). orientation(mc1gt_3,146,183). coords(mc1gt_3,114,82,183).  visible(mc1gt_3,183). happensAt(walking(mc1gt_3),183).
orientation(mc1gt_0,147,184). coords(mc1gt_0,123,112,184).  visible(mc1gt_0,184). happensAt(walking(mc1gt_0),184). orientation(mc1gt_3,146,184). coords(mc1gt_3,115,84,184).  visible(mc1gt_3,184). happensAt(walking(mc1gt_3),184).
orientation(mc1gt_0,147,185). coords(mc1gt_0,123,113,185).  visible(mc1gt_0,185). happensAt(walking(mc1gt_0),185). orientation(mc1gt_3,146,185). coords(mc1gt_3,115,84,185).  visible(mc1gt_3,185). happensAt(walking(mc1gt_3),185).
orientation(mc1gt_0,147,186). coords(mc1gt_0,123,114,186).  visible(mc1gt_0,186). happensAt(walking(mc1gt_0),186). orientation(mc1gt_3,146,186). coords(mc1gt_3,117,84,186).  visible(mc1gt_3,186). happensAt(walking(mc1gt_3),186).
orientation(mc1gt_0,147,187). coords(mc1gt_0,124,114,187).  visible(mc1gt_0,187). happensAt(walking(mc1gt_0),187). orientation(mc1gt_3,146,187). coords(mc1gt_3,119,85,187).  visible(mc1gt_3,187). happensAt(walking(mc1gt_3),187).
orientation(mc1gt_0,147,188). coords(mc1gt_0,126,115,188).  visible(mc1gt_0,188). happensAt(walking(mc1gt_0),188). orientation(mc1gt_3,146,188). coords(mc1gt_3,119,83,188).  visible(mc1gt_3,188). happensAt(walking(mc1gt_3),188).
orientation(mc1gt_0,147,189). coords(mc1gt_0,127,115,189).  visible(mc1gt_0,189). happensAt(walking(mc1gt_0),189). orientation(mc1gt_3,146,189). coords(mc1gt_3,118,84,189).  visible(mc1gt_3,189). happensAt(walking(mc1gt_3),189).
orientation(mc1gt_0,147,190). coords(mc1gt_0,128,116,190).  visible(mc1gt_0,190). happensAt(walking(mc1gt_0),190). orientation(mc1gt_3,146,190). coords(mc1gt_3,117,82,190).  visible(mc1gt_3,190). happensAt(walking(mc1gt_3),190).
orientation(mc1gt_0,147,191). coords(mc1gt_0,129,116,191).  visible(mc1gt_0,191). happensAt(walking(mc1gt_0),191). orientation(mc1gt_3,146,191). coords(mc1gt_3,118,82,191).  visible(mc1gt_3,191). happensAt(walking(mc1gt_3),191).
orientation(mc1gt_0,147,192). coords(mc1gt_0,130,116,192).  visible(mc1gt_0,192). happensAt(walking(mc1gt_0),192). orientation(mc1gt_3,146,192). coords(mc1gt_3,118,82,192).  visible(mc1gt_3,192). happensAt(walking(mc1gt_3),192).
orientation(mc1gt_0,147,193). coords(mc1gt_0,130,117,193).  visible(mc1gt_0,193). happensAt(walking(mc1gt_0),193). orientation(mc1gt_3,146,193). coords(mc1gt_3,118,82,193).  visible(mc1gt_3,193). happensAt(walking(mc1gt_3),193).
orientation(mc1gt_0,147,194). coords(mc1gt_0,130,117,194).  visible(mc1gt_0,194). happensAt(walking(mc1gt_0),194). orientation(mc1gt_3,146,194). coords(mc1gt_3,119,82,194).  visible(mc1gt_3,194). happensAt(walking(mc1gt_3),194).
orientation(mc1gt_0,147,195). coords(mc1gt_0,131,117,195).  visible(mc1gt_0,195). happensAt(walking(mc1gt_0),195). orientation(mc1gt_3,146,195). coords(mc1gt_3,120,82,195).  visible(mc1gt_3,195). happensAt(walking(mc1gt_3),195).
orientation(mc1gt_0,147,196). coords(mc1gt_0,131,117,196).  visible(mc1gt_0,196). happensAt(walking(mc1gt_0),196). orientation(mc1gt_3,146,196). coords(mc1gt_3,120,82,196).  visible(mc1gt_3,196). happensAt(walking(mc1gt_3),196).
orientation(mc1gt_0,147,197). coords(mc1gt_0,131,117,197).  visible(mc1gt_0,197). happensAt(walking(mc1gt_0),197). orientation(mc1gt_3,146,197). coords(mc1gt_3,121,82,197).  visible(mc1gt_3,197). happensAt(walking(mc1gt_3),197).
orientation(mc1gt_0,147,198). coords(mc1gt_0,131,117,198).  visible(mc1gt_0,198). happensAt(walking(mc1gt_0),198). orientation(mc1gt_3,146,198). coords(mc1gt_3,122,82,198).  visible(mc1gt_3,198). happensAt(walking(mc1gt_3),198).
orientation(mc1gt_0,147,199). coords(mc1gt_0,132,117,199).  visible(mc1gt_0,199). happensAt(walking(mc1gt_0),199). orientation(mc1gt_3,146,199). coords(mc1gt_3,123,82,199).  visible(mc1gt_3,199). happensAt(walking(mc1gt_3),199).
orientation(mc1gt_0,147,200). coords(mc1gt_0,134,117,200).  visible(mc1gt_0,200). happensAt(walking(mc1gt_0),200). orientation(mc1gt_3,146,200). coords(mc1gt_3,126,83,200).  visible(mc1gt_3,200). happensAt(walking(mc1gt_3),200).
orientation(mc1gt_0,147,201). coords(mc1gt_0,136,118,201).  visible(mc1gt_0,201). happensAt(walking(mc1gt_0),201). orientation(mc1gt_3,146,201). coords(mc1gt_3,127,83,201).  visible(mc1gt_3,201). happensAt(walking(mc1gt_3),201).
orientation(mc1gt_0,147,202). coords(mc1gt_0,138,118,202).  visible(mc1gt_0,202). happensAt(walking(mc1gt_0),202). orientation(mc1gt_3,146,202). coords(mc1gt_3,129,83,202).  visible(mc1gt_3,202). happensAt(walking(mc1gt_3),202).
orientation(mc1gt_0,147,203). coords(mc1gt_0,139,118,203).  visible(mc1gt_0,203). happensAt(walking(mc1gt_0),203). orientation(mc1gt_3,146,203). coords(mc1gt_3,131,83,203).  visible(mc1gt_3,203). happensAt(walking(mc1gt_3),203).
orientation(mc1gt_0,147,204). coords(mc1gt_0,140,118,204).  visible(mc1gt_0,204). happensAt(walking(mc1gt_0),204). orientation(mc1gt_3,146,204). coords(mc1gt_3,133,84,204).  visible(mc1gt_3,204). happensAt(walking(mc1gt_3),204).
#show holdsAt/2.