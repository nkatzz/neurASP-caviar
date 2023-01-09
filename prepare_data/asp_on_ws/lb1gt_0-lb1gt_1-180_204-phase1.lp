
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

	orientation(lb1gt_0,136,180). coords(lb1gt_0,161,38,180).  visible(lb1gt_0,180). happensAt(walking(lb1gt_0),180). orientation(lb1gt_1,151,180). coords(lb1gt_1,108,149,180).  visible(lb1gt_1,180). happensAt(walking(lb1gt_1),180).
orientation(lb1gt_0,136,181). coords(lb1gt_0,162,38,181).  visible(lb1gt_0,181). happensAt(walking(lb1gt_0),181). orientation(lb1gt_1,151,181). coords(lb1gt_1,109,149,181).  visible(lb1gt_1,181). happensAt(walking(lb1gt_1),181).
orientation(lb1gt_0,136,182). coords(lb1gt_0,162,38,182).  visible(lb1gt_0,182). happensAt(walking(lb1gt_0),182). orientation(lb1gt_1,151,182). coords(lb1gt_1,110,150,182).  visible(lb1gt_1,182). happensAt(walking(lb1gt_1),182).
orientation(lb1gt_0,136,183). coords(lb1gt_0,162,38,183).  visible(lb1gt_0,183). happensAt(walking(lb1gt_0),183). orientation(lb1gt_1,151,183). coords(lb1gt_1,110,150,183).  visible(lb1gt_1,183). happensAt(walking(lb1gt_1),183).
orientation(lb1gt_0,136,184). coords(lb1gt_0,163,38,184).  visible(lb1gt_0,184). happensAt(walking(lb1gt_0),184). orientation(lb1gt_1,151,184). coords(lb1gt_1,111,150,184).  visible(lb1gt_1,184). happensAt(walking(lb1gt_1),184).
orientation(lb1gt_0,136,185). coords(lb1gt_0,163,37,185).  visible(lb1gt_0,185). happensAt(walking(lb1gt_0),185). orientation(lb1gt_1,151,185). coords(lb1gt_1,111,150,185).  visible(lb1gt_1,185). happensAt(walking(lb1gt_1),185).
orientation(lb1gt_0,136,186). coords(lb1gt_0,164,37,186).  visible(lb1gt_0,186). happensAt(walking(lb1gt_0),186). orientation(lb1gt_1,151,186). coords(lb1gt_1,111,150,186).  visible(lb1gt_1,186). happensAt(walking(lb1gt_1),186).
orientation(lb1gt_0,136,187). coords(lb1gt_0,165,35,187).  visible(lb1gt_0,187). happensAt(walking(lb1gt_0),187). orientation(lb1gt_1,151,187). coords(lb1gt_1,111,150,187).  visible(lb1gt_1,187). happensAt(walking(lb1gt_1),187).
orientation(lb1gt_0,136,188). coords(lb1gt_0,165,35,188).  visible(lb1gt_0,188). happensAt(walking(lb1gt_0),188). orientation(lb1gt_1,151,188). coords(lb1gt_1,111,150,188).  visible(lb1gt_1,188). happensAt(walking(lb1gt_1),188).
orientation(lb1gt_0,136,189). coords(lb1gt_0,165,34,189).  visible(lb1gt_0,189). happensAt(walking(lb1gt_0),189). orientation(lb1gt_1,151,189). coords(lb1gt_1,112,150,189).  visible(lb1gt_1,189). happensAt(walking(lb1gt_1),189).
orientation(lb1gt_0,136,190). coords(lb1gt_0,166,34,190).  visible(lb1gt_0,190). happensAt(walking(lb1gt_0),190). orientation(lb1gt_1,151,190). coords(lb1gt_1,112,150,190).  visible(lb1gt_1,190). happensAt(walking(lb1gt_1),190).
orientation(lb1gt_0,136,191). coords(lb1gt_0,166,34,191).  visible(lb1gt_0,191). happensAt(walking(lb1gt_0),191). orientation(lb1gt_1,151,191). coords(lb1gt_1,112,151,191).  visible(lb1gt_1,191). happensAt(walking(lb1gt_1),191).
orientation(lb1gt_0,136,192). coords(lb1gt_0,166,33,192).  visible(lb1gt_0,192). happensAt(walking(lb1gt_0),192). orientation(lb1gt_1,151,192). coords(lb1gt_1,112,151,192).  visible(lb1gt_1,192). happensAt(walking(lb1gt_1),192).
orientation(lb1gt_0,136,193). coords(lb1gt_0,166,33,193).  visible(lb1gt_0,193). happensAt(walking(lb1gt_0),193). orientation(lb1gt_1,151,193). coords(lb1gt_1,112,152,193).  visible(lb1gt_1,193). happensAt(walking(lb1gt_1),193).
orientation(lb1gt_0,136,194). coords(lb1gt_0,166,33,194).  visible(lb1gt_0,194). happensAt(walking(lb1gt_0),194). orientation(lb1gt_1,151,194). coords(lb1gt_1,113,153,194).  visible(lb1gt_1,194). happensAt(walking(lb1gt_1),194).
orientation(lb1gt_0,136,195). coords(lb1gt_0,166,33,195).  visible(lb1gt_0,195). happensAt(walking(lb1gt_0),195). orientation(lb1gt_1,151,195). coords(lb1gt_1,113,154,195).  visible(lb1gt_1,195). happensAt(walking(lb1gt_1),195).
orientation(lb1gt_0,136,196). coords(lb1gt_0,166,32,196).  visible(lb1gt_0,196). happensAt(walking(lb1gt_0),196). orientation(lb1gt_1,151,196). coords(lb1gt_1,115,155,196).  visible(lb1gt_1,196). happensAt(walking(lb1gt_1),196).
orientation(lb1gt_0,136,197). coords(lb1gt_0,167,32,197).  visible(lb1gt_0,197). happensAt(walking(lb1gt_0),197). orientation(lb1gt_1,151,197). coords(lb1gt_1,116,156,197).  visible(lb1gt_1,197). happensAt(walking(lb1gt_1),197).
orientation(lb1gt_0,136,198). coords(lb1gt_0,167,32,198).  visible(lb1gt_0,198). happensAt(walking(lb1gt_0),198). orientation(lb1gt_1,151,198). coords(lb1gt_1,118,156,198).  visible(lb1gt_1,198). happensAt(walking(lb1gt_1),198).
orientation(lb1gt_0,136,199). coords(lb1gt_0,168,31,199).  visible(lb1gt_0,199). happensAt(walking(lb1gt_0),199). orientation(lb1gt_1,151,199). coords(lb1gt_1,118,156,199).  visible(lb1gt_1,199). happensAt(walking(lb1gt_1),199).
orientation(lb1gt_0,136,200). coords(lb1gt_0,168,31,200).  visible(lb1gt_0,200). happensAt(walking(lb1gt_0),200). orientation(lb1gt_1,151,200). coords(lb1gt_1,119,157,200).  visible(lb1gt_1,200). happensAt(walking(lb1gt_1),200).
orientation(lb1gt_0,136,201). coords(lb1gt_0,168,31,201).  visible(lb1gt_0,201). happensAt(walking(lb1gt_0),201). orientation(lb1gt_1,151,201). coords(lb1gt_1,120,157,201).  visible(lb1gt_1,201). happensAt(walking(lb1gt_1),201).
orientation(lb1gt_0,136,202). coords(lb1gt_0,168,31,202).  visible(lb1gt_0,202). happensAt(walking(lb1gt_0),202). orientation(lb1gt_1,151,202). coords(lb1gt_1,120,157,202).  visible(lb1gt_1,202). happensAt(walking(lb1gt_1),202).
orientation(lb1gt_0,136,203). coords(lb1gt_0,169,31,203).  visible(lb1gt_0,203). happensAt(walking(lb1gt_0),203). orientation(lb1gt_1,151,203). coords(lb1gt_1,121,158,203).  visible(lb1gt_1,203). happensAt(walking(lb1gt_1),203).
orientation(lb1gt_0,136,204). coords(lb1gt_0,169,30,204).  visible(lb1gt_0,204). happensAt(walking(lb1gt_0),204). orientation(lb1gt_1,151,204). coords(lb1gt_1,121,158,204).  visible(lb1gt_1,204). happensAt(walking(lb1gt_1),204).
#show holdsAt/2.