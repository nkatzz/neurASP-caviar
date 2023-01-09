
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

	orientation(br2gt_1,149,182). coords(br2gt_1,240,198,182).  visible(br2gt_1,182). happensAt(walking(br2gt_1),182). orientation(br2gt_2,144,182). coords(br2gt_2,76,62,182).  visible(br2gt_2,182). happensAt(walking(br2gt_2),182).
orientation(br2gt_1,149,183). coords(br2gt_1,240,199,183).  visible(br2gt_1,183). happensAt(walking(br2gt_1),183). orientation(br2gt_2,144,183). coords(br2gt_2,76,63,183).  visible(br2gt_2,183). happensAt(walking(br2gt_2),183).
orientation(br2gt_1,160,184). coords(br2gt_1,244,201,184).  visible(br2gt_1,184). happensAt(walking(br2gt_1),184). orientation(br2gt_2,144,184). coords(br2gt_2,76,64,184).  visible(br2gt_2,184). happensAt(walking(br2gt_2),184).
orientation(br2gt_1,165,185). coords(br2gt_1,247,204,185).  visible(br2gt_1,185). happensAt(walking(br2gt_1),185). orientation(br2gt_2,144,185). coords(br2gt_2,76,64,185).  visible(br2gt_2,185). happensAt(walking(br2gt_2),185).
orientation(br2gt_1,171,186). coords(br2gt_1,251,206,186).  visible(br2gt_1,186). happensAt(walking(br2gt_1),186). orientation(br2gt_2,144,186). coords(br2gt_2,76,66,186).  visible(br2gt_2,186). happensAt(walking(br2gt_2),186).
orientation(br2gt_1,176,187). coords(br2gt_1,254,208,187).  visible(br2gt_1,187). happensAt(walking(br2gt_1),187). orientation(br2gt_2,144,187). coords(br2gt_2,76,67,187).  visible(br2gt_2,187). happensAt(walking(br2gt_2),187).
orientation(br2gt_1,9,188). coords(br2gt_1,257,211,188).  visible(br2gt_1,188). happensAt(walking(br2gt_1),188). orientation(br2gt_2,144,188). coords(br2gt_2,76,67,188).  visible(br2gt_2,188). happensAt(walking(br2gt_2),188).
orientation(br2gt_1,9,189). coords(br2gt_1,259,213,189).  visible(br2gt_1,189). happensAt(walking(br2gt_1),189). orientation(br2gt_2,144,189). coords(br2gt_2,76,67,189).  visible(br2gt_2,189). happensAt(walking(br2gt_2),189).
orientation(br2gt_1,9,190). coords(br2gt_1,262,216,190).  visible(br2gt_1,190). happensAt(walking(br2gt_1),190). orientation(br2gt_2,144,190). coords(br2gt_2,76,67,190).  visible(br2gt_2,190). happensAt(walking(br2gt_2),190).
orientation(br2gt_1,18,191). coords(br2gt_1,263,218,191).  visible(br2gt_1,191). happensAt(walking(br2gt_1),191). orientation(br2gt_2,144,191). coords(br2gt_2,76,67,191).  visible(br2gt_2,191). happensAt(walking(br2gt_2),191).
orientation(br2gt_1,18,192). coords(br2gt_1,264,221,192).  visible(br2gt_1,192). happensAt(walking(br2gt_1),192). orientation(br2gt_2,144,192). coords(br2gt_2,76,67,192).  visible(br2gt_2,192). happensAt(walking(br2gt_2),192).
orientation(br2gt_1,18,193). coords(br2gt_1,264,223,193).  visible(br2gt_1,193). happensAt(walking(br2gt_1),193). orientation(br2gt_2,144,193). coords(br2gt_2,76,67,193).  visible(br2gt_2,193). happensAt(walking(br2gt_2),193).
orientation(br2gt_1,18,194). coords(br2gt_1,265,225,194).  visible(br2gt_1,194). happensAt(walking(br2gt_1),194). orientation(br2gt_2,144,194). coords(br2gt_2,75,67,194).  visible(br2gt_2,194). happensAt(walking(br2gt_2),194).
orientation(br2gt_1,29,195). coords(br2gt_1,266,230,195).  visible(br2gt_1,195). happensAt(walking(br2gt_1),195). orientation(br2gt_2,144,195). coords(br2gt_2,75,67,195).  visible(br2gt_2,195). happensAt(walking(br2gt_2),195).
orientation(br2gt_1,45,196). coords(br2gt_1,267,234,196).  visible(br2gt_1,196). happensAt(walking(br2gt_1),196). orientation(br2gt_2,144,196). coords(br2gt_2,75,67,196).  visible(br2gt_2,196). happensAt(walking(br2gt_2),196).
orientation(br2gt_1,54,197). coords(br2gt_1,269,238,197).  visible(br2gt_1,197). happensAt(walking(br2gt_1),197). orientation(br2gt_2,144,197). coords(br2gt_2,76,67,197).  visible(br2gt_2,197). happensAt(walking(br2gt_2),197).
orientation(br2gt_1,54,198). coords(br2gt_1,272,241,198).  visible(br2gt_1,198). happensAt(walking(br2gt_1),198). orientation(br2gt_2,144,198). coords(br2gt_2,76,67,198).  visible(br2gt_2,198). happensAt(walking(br2gt_2),198).
orientation(br2gt_1,54,199). coords(br2gt_1,275,242,199).  visible(br2gt_1,199). happensAt(walking(br2gt_1),199). orientation(br2gt_2,144,199). coords(br2gt_2,76,67,199).  visible(br2gt_2,199). happensAt(walking(br2gt_2),199).
orientation(br2gt_1,54,200). coords(br2gt_1,276,244,200).  visible(br2gt_1,200). happensAt(walking(br2gt_1),200). orientation(br2gt_2,144,200). coords(br2gt_2,76,67,200).  visible(br2gt_2,200). happensAt(walking(br2gt_2),200).
orientation(br2gt_1,62,201). coords(br2gt_1,276,247,201).  visible(br2gt_1,201). happensAt(walking(br2gt_1),201). orientation(br2gt_2,144,201). coords(br2gt_2,76,68,201).  visible(br2gt_2,201). happensAt(walking(br2gt_2),201).
orientation(br2gt_1,69,202). coords(br2gt_1,279,249,202).  visible(br2gt_1,202). happensAt(walking(br2gt_1),202). orientation(br2gt_2,144,202). coords(br2gt_2,76,68,202).  visible(br2gt_2,202). happensAt(walking(br2gt_2),202).
orientation(br2gt_1,78,203). coords(br2gt_1,279,250,203).  visible(br2gt_1,203). happensAt(walking(br2gt_1),203). orientation(br2gt_2,144,203). coords(br2gt_2,76,69,203).  visible(br2gt_2,203). happensAt(walking(br2gt_2),203).
orientation(br2gt_1,78,204). coords(br2gt_1,281,252,204).  visible(br2gt_1,204). happensAt(walking(br2gt_1),204). orientation(br2gt_2,144,204). coords(br2gt_2,76,69,204).  visible(br2gt_2,204). happensAt(walking(br2gt_2),204).
orientation(br2gt_1,78,205). coords(br2gt_1,282,255,205).  visible(br2gt_1,205). happensAt(walking(br2gt_1),205). orientation(br2gt_2,144,205). coords(br2gt_2,76,69,205).  visible(br2gt_2,205). happensAt(walking(br2gt_2),205).
orientation(br2gt_1,78,206). coords(br2gt_1,284,258,206).  visible(br2gt_1,206). happensAt(walking(br2gt_1),206). orientation(br2gt_2,144,206). coords(br2gt_2,76,69,206).  visible(br2gt_2,206). happensAt(walking(br2gt_2),206).
#show holdsAt/2.