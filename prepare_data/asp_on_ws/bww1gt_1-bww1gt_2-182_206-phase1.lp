
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

	orientation(bww1gt_1,14,182). coords(bww1gt_1,137,278,182).  visible(bww1gt_1,182). happensAt(walking(bww1gt_1),182). orientation(bww1gt_2,167,182). coords(bww1gt_2,29,163,182).  happensAt(appear(bww1gt_2),182). happensAt(walking(bww1gt_2),182).
orientation(bww1gt_1,14,183). coords(bww1gt_1,136,278,183).  visible(bww1gt_1,183). happensAt(walking(bww1gt_1),183). orientation(bww1gt_2,167,183). coords(bww1gt_2,29,163,183).  visible(bww1gt_2,183). happensAt(walking(bww1gt_2),183).
orientation(bww1gt_1,14,184). coords(bww1gt_1,136,278,184).  visible(bww1gt_1,184). happensAt(walking(bww1gt_1),184). orientation(bww1gt_2,167,184). coords(bww1gt_2,29,163,184).  visible(bww1gt_2,184). happensAt(walking(bww1gt_2),184).
orientation(bww1gt_1,14,185). coords(bww1gt_1,133,278,185).  visible(bww1gt_1,185). happensAt(walking(bww1gt_1),185). orientation(bww1gt_2,167,185). coords(bww1gt_2,29,163,185).  visible(bww1gt_2,185). happensAt(walking(bww1gt_2),185).
orientation(bww1gt_1,14,186). coords(bww1gt_1,131,278,186).  visible(bww1gt_1,186). happensAt(walking(bww1gt_1),186). orientation(bww1gt_2,167,186). coords(bww1gt_2,29,163,186).  visible(bww1gt_2,186). happensAt(walking(bww1gt_2),186).
orientation(bww1gt_1,14,187). coords(bww1gt_1,129,277,187).  visible(bww1gt_1,187). happensAt(walking(bww1gt_1),187). orientation(bww1gt_2,167,187). coords(bww1gt_2,29,163,187).  visible(bww1gt_2,187). happensAt(walking(bww1gt_2),187).
orientation(bww1gt_1,14,188). coords(bww1gt_1,126,277,188).  visible(bww1gt_1,188). happensAt(walking(bww1gt_1),188). orientation(bww1gt_2,167,188). coords(bww1gt_2,30,163,188).  visible(bww1gt_2,188). happensAt(walking(bww1gt_2),188).
orientation(bww1gt_1,14,189). coords(bww1gt_1,125,277,189).  visible(bww1gt_1,189). happensAt(walking(bww1gt_1),189). orientation(bww1gt_2,167,189). coords(bww1gt_2,30,163,189).  visible(bww1gt_2,189). happensAt(walking(bww1gt_2),189).
orientation(bww1gt_1,2,190). coords(bww1gt_1,125,276,190).  visible(bww1gt_1,190). happensAt(walking(bww1gt_1),190). orientation(bww1gt_2,167,190). coords(bww1gt_2,30,164,190).  visible(bww1gt_2,190). happensAt(walking(bww1gt_2),190).
orientation(bww1gt_1,0,191). coords(bww1gt_1,124,276,191).  visible(bww1gt_1,191). happensAt(walking(bww1gt_1),191). orientation(bww1gt_2,167,191). coords(bww1gt_2,30,164,191).  visible(bww1gt_2,191). happensAt(walking(bww1gt_2),191).
orientation(bww1gt_1,0,192). coords(bww1gt_1,123,275,192).  visible(bww1gt_1,192). happensAt(walking(bww1gt_1),192). orientation(bww1gt_2,167,192). coords(bww1gt_2,30,164,192).  visible(bww1gt_2,192). happensAt(walking(bww1gt_2),192).
orientation(bww1gt_1,179,193). coords(bww1gt_1,122,273,193).  visible(bww1gt_1,193). happensAt(walking(bww1gt_1),193). orientation(bww1gt_2,167,193). coords(bww1gt_2,30,164,193).  visible(bww1gt_2,193). happensAt(walking(bww1gt_2),193).
orientation(bww1gt_1,176,194). coords(bww1gt_1,122,273,194).  visible(bww1gt_1,194). happensAt(walking(bww1gt_1),194). orientation(bww1gt_2,167,194). coords(bww1gt_2,31,164,194).  visible(bww1gt_2,194). happensAt(walking(bww1gt_2),194).
orientation(bww1gt_1,175,195). coords(bww1gt_1,122,273,195).  visible(bww1gt_1,195). happensAt(walking(bww1gt_1),195). orientation(bww1gt_2,167,195). coords(bww1gt_2,31,164,195).  visible(bww1gt_2,195). happensAt(walking(bww1gt_2),195).
orientation(bww1gt_1,1,196). coords(bww1gt_1,121,273,196).  visible(bww1gt_1,196). happensAt(walking(bww1gt_1),196). orientation(bww1gt_2,167,196). coords(bww1gt_2,31,164,196).  visible(bww1gt_2,196). happensAt(walking(bww1gt_2),196).
orientation(bww1gt_1,176,197). coords(bww1gt_1,120,273,197).  visible(bww1gt_1,197). happensAt(walking(bww1gt_1),197). orientation(bww1gt_2,167,197). coords(bww1gt_2,31,164,197).  visible(bww1gt_2,197). happensAt(walking(bww1gt_2),197).
orientation(bww1gt_1,179,198). coords(bww1gt_1,118,273,198).  visible(bww1gt_1,198). happensAt(walking(bww1gt_1),198). orientation(bww1gt_2,167,198). coords(bww1gt_2,31,164,198).  visible(bww1gt_2,198). happensAt(walking(bww1gt_2),198).
orientation(bww1gt_1,178,199). coords(bww1gt_1,117,272,199).  visible(bww1gt_1,199). happensAt(walking(bww1gt_1),199). orientation(bww1gt_2,167,199). coords(bww1gt_2,31,164,199).  visible(bww1gt_2,199). happensAt(walking(bww1gt_2),199).
orientation(bww1gt_1,177,200). coords(bww1gt_1,116,271,200).  visible(bww1gt_1,200). happensAt(walking(bww1gt_1),200). orientation(bww1gt_2,167,200). coords(bww1gt_2,31,164,200).  visible(bww1gt_2,200). happensAt(walking(bww1gt_2),200).
orientation(bww1gt_1,0,201). coords(bww1gt_1,115,271,201).  visible(bww1gt_1,201). happensAt(walking(bww1gt_1),201). orientation(bww1gt_2,167,201). coords(bww1gt_2,32,164,201).  visible(bww1gt_2,201). happensAt(walking(bww1gt_2),201).
orientation(bww1gt_1,174,202). coords(bww1gt_1,115,271,202).  visible(bww1gt_1,202). happensAt(walking(bww1gt_1),202). orientation(bww1gt_2,167,202). coords(bww1gt_2,32,164,202).  visible(bww1gt_2,202). happensAt(walking(bww1gt_2),202).
orientation(bww1gt_1,178,203). coords(bww1gt_1,114,271,203).  visible(bww1gt_1,203). happensAt(walking(bww1gt_1),203). orientation(bww1gt_2,167,203). coords(bww1gt_2,32,164,203).  visible(bww1gt_2,203). happensAt(walking(bww1gt_2),203).
orientation(bww1gt_1,177,204). coords(bww1gt_1,113,271,204).  visible(bww1gt_1,204). happensAt(walking(bww1gt_1),204). orientation(bww1gt_2,167,204). coords(bww1gt_2,32,164,204).  visible(bww1gt_2,204). happensAt(walking(bww1gt_2),204).
orientation(bww1gt_1,177,205). coords(bww1gt_1,111,270,205).  visible(bww1gt_1,205). happensAt(walking(bww1gt_1),205). orientation(bww1gt_2,167,205). coords(bww1gt_2,33,165,205).  visible(bww1gt_2,205). happensAt(walking(bww1gt_2),205).
orientation(bww1gt_1,173,206). coords(bww1gt_1,110,270,206).  visible(bww1gt_1,206). happensAt(walking(bww1gt_1),206). orientation(bww1gt_2,167,206). coords(bww1gt_2,33,165,206).  visible(bww1gt_2,206). happensAt(walking(bww1gt_2),206).
#show holdsAt/2.