
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

	orientation(mwt2gt_1,137,181). coords(mwt2gt_1,162,37,181).  visible(mwt2gt_1,181). happensAt(walking(mwt2gt_1),181). orientation(mwt2gt_2,59,181). coords(mwt2gt_2,260,277,181).  visible(mwt2gt_2,181). happensAt(walking(mwt2gt_2),181).
orientation(mwt2gt_1,137,182). coords(mwt2gt_1,162,37,182).  visible(mwt2gt_1,182). happensAt(walking(mwt2gt_1),182). orientation(mwt2gt_2,60,182). coords(mwt2gt_2,255,272,182).  visible(mwt2gt_2,182). happensAt(walking(mwt2gt_2),182).
orientation(mwt2gt_1,137,183). coords(mwt2gt_1,162,37,183).  visible(mwt2gt_1,183). happensAt(walking(mwt2gt_1),183). orientation(mwt2gt_2,42,183). coords(mwt2gt_2,254,272,183).  visible(mwt2gt_2,183). happensAt(walking(mwt2gt_2),183).
orientation(mwt2gt_1,137,184). coords(mwt2gt_1,162,37,184).  visible(mwt2gt_1,184). happensAt(walking(mwt2gt_1),184). orientation(mwt2gt_2,46,184). coords(mwt2gt_2,254,271,184).  visible(mwt2gt_2,184). happensAt(walking(mwt2gt_2),184).
orientation(mwt2gt_1,137,185). coords(mwt2gt_1,161,40,185).  visible(mwt2gt_1,185). happensAt(walking(mwt2gt_1),185). orientation(mwt2gt_2,34,185). coords(mwt2gt_2,253,271,185).  visible(mwt2gt_2,185). happensAt(walking(mwt2gt_2),185).
orientation(mwt2gt_1,137,186). coords(mwt2gt_1,161,40,186).  visible(mwt2gt_1,186). happensAt(walking(mwt2gt_1),186). orientation(mwt2gt_2,29,186). coords(mwt2gt_2,253,271,186).  visible(mwt2gt_2,186). happensAt(walking(mwt2gt_2),186).
orientation(mwt2gt_1,137,187). coords(mwt2gt_1,160,40,187).  visible(mwt2gt_1,187). happensAt(walking(mwt2gt_1),187). orientation(mwt2gt_2,29,187). coords(mwt2gt_2,252,269,187).  visible(mwt2gt_2,187). happensAt(walking(mwt2gt_2),187).
orientation(mwt2gt_1,137,188). coords(mwt2gt_1,160,40,188).  visible(mwt2gt_1,188). happensAt(walking(mwt2gt_1),188). orientation(mwt2gt_2,29,188). coords(mwt2gt_2,251,267,188).  visible(mwt2gt_2,188). happensAt(walking(mwt2gt_2),188).
orientation(mwt2gt_1,137,189). coords(mwt2gt_1,160,41,189).  visible(mwt2gt_1,189). happensAt(walking(mwt2gt_1),189). orientation(mwt2gt_2,29,189). coords(mwt2gt_2,250,266,189).  visible(mwt2gt_2,189). happensAt(walking(mwt2gt_2),189).
orientation(mwt2gt_1,137,190). coords(mwt2gt_1,160,41,190).  visible(mwt2gt_1,190). happensAt(walking(mwt2gt_1),190). orientation(mwt2gt_2,25,190). coords(mwt2gt_2,249,265,190).  visible(mwt2gt_2,190). happensAt(walking(mwt2gt_2),190).
orientation(mwt2gt_1,141,191). coords(mwt2gt_1,158,41,191).  visible(mwt2gt_1,191). happensAt(walking(mwt2gt_1),191). orientation(mwt2gt_2,25,191). coords(mwt2gt_2,249,263,191).  visible(mwt2gt_2,191). happensAt(walking(mwt2gt_2),191).
orientation(mwt2gt_1,141,192). coords(mwt2gt_1,158,41,192).  visible(mwt2gt_1,192). happensAt(walking(mwt2gt_1),192). orientation(mwt2gt_2,25,192). coords(mwt2gt_2,247,261,192).  visible(mwt2gt_2,192). happensAt(walking(mwt2gt_2),192).
orientation(mwt2gt_1,141,193). coords(mwt2gt_1,156,41,193).  visible(mwt2gt_1,193). happensAt(walking(mwt2gt_1),193). orientation(mwt2gt_2,25,193). coords(mwt2gt_2,247,257,193).  visible(mwt2gt_2,193). happensAt(walking(mwt2gt_2),193).
orientation(mwt2gt_1,141,194). coords(mwt2gt_1,155,41,194).  visible(mwt2gt_1,194). happensAt(walking(mwt2gt_1),194). orientation(mwt2gt_2,25,194). coords(mwt2gt_2,246,252,194).  visible(mwt2gt_2,194). happensAt(walking(mwt2gt_2),194).
orientation(mwt2gt_1,141,195). coords(mwt2gt_1,154,41,195).  visible(mwt2gt_1,195). happensAt(walking(mwt2gt_1),195). orientation(mwt2gt_2,25,195). coords(mwt2gt_2,245,248,195).  visible(mwt2gt_2,195). happensAt(walking(mwt2gt_2),195).
orientation(mwt2gt_1,141,196). coords(mwt2gt_1,154,41,196).  visible(mwt2gt_1,196). happensAt(walking(mwt2gt_1),196). orientation(mwt2gt_2,25,196). coords(mwt2gt_2,245,245,196).  visible(mwt2gt_2,196). happensAt(walking(mwt2gt_2),196).
orientation(mwt2gt_1,141,197). coords(mwt2gt_1,154,41,197).  visible(mwt2gt_1,197). happensAt(walking(mwt2gt_1),197). orientation(mwt2gt_2,19,197). coords(mwt2gt_2,245,245,197).  visible(mwt2gt_2,197). happensAt(walking(mwt2gt_2),197).
orientation(mwt2gt_1,141,198). coords(mwt2gt_1,153,42,198).  visible(mwt2gt_1,198). happensAt(walking(mwt2gt_1),198). orientation(mwt2gt_2,15,198). coords(mwt2gt_2,244,244,198).  visible(mwt2gt_2,198). happensAt(walking(mwt2gt_2),198).
orientation(mwt2gt_1,141,199). coords(mwt2gt_1,153,43,199).  visible(mwt2gt_1,199). happensAt(walking(mwt2gt_1),199). orientation(mwt2gt_2,15,199). coords(mwt2gt_2,244,244,199).  visible(mwt2gt_2,199). happensAt(walking(mwt2gt_2),199).
orientation(mwt2gt_1,141,200). coords(mwt2gt_1,153,45,200).  visible(mwt2gt_1,200). happensAt(walking(mwt2gt_1),200). orientation(mwt2gt_2,15,200). coords(mwt2gt_2,243,243,200).  visible(mwt2gt_2,200). happensAt(walking(mwt2gt_2),200).
orientation(mwt2gt_1,141,201). coords(mwt2gt_1,153,46,201).  visible(mwt2gt_1,201). happensAt(walking(mwt2gt_1),201). orientation(mwt2gt_2,0,201). coords(mwt2gt_2,243,241,201).  visible(mwt2gt_2,201). happensAt(walking(mwt2gt_2),201).
orientation(mwt2gt_1,141,202). coords(mwt2gt_1,153,47,202).  visible(mwt2gt_1,202). happensAt(walking(mwt2gt_1),202). orientation(mwt2gt_2,0,202). coords(mwt2gt_2,242,239,202).  visible(mwt2gt_2,202). happensAt(walking(mwt2gt_2),202).
orientation(mwt2gt_1,141,203). coords(mwt2gt_1,152,48,203).  visible(mwt2gt_1,203). happensAt(walking(mwt2gt_1),203). orientation(mwt2gt_2,174,203). coords(mwt2gt_2,242,237,203).  visible(mwt2gt_2,203). happensAt(walking(mwt2gt_2),203).
orientation(mwt2gt_1,141,204). coords(mwt2gt_1,152,48,204).  visible(mwt2gt_1,204). happensAt(walking(mwt2gt_1),204). orientation(mwt2gt_2,174,204). coords(mwt2gt_2,242,234,204).  visible(mwt2gt_2,204). happensAt(walking(mwt2gt_2),204).
orientation(mwt2gt_1,141,205). coords(mwt2gt_1,152,49,205).  visible(mwt2gt_1,205). happensAt(walking(mwt2gt_1),205). orientation(mwt2gt_2,177,205). coords(mwt2gt_2,241,231,205).  visible(mwt2gt_2,205). happensAt(walking(mwt2gt_2),205).
#show holdsAt/2.