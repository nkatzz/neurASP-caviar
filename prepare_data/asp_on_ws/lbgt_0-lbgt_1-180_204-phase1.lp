
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

	orientation(lbgt_0,171,180). coords(lbgt_0,213,194,180).  visible(lbgt_0,180). happensAt(walking(lbgt_0),180). orientation(lbgt_1,152,180). coords(lbgt_1,79,98,180).  visible(lbgt_1,180). happensAt(walking(lbgt_1),180).
orientation(lbgt_0,168,181). coords(lbgt_0,212,193,181).  visible(lbgt_0,181). happensAt(walking(lbgt_0),181). orientation(lbgt_1,152,181). coords(lbgt_1,79,98,181).  visible(lbgt_1,181). happensAt(walking(lbgt_1),181).
orientation(lbgt_0,168,182). coords(lbgt_0,210,194,182).  visible(lbgt_0,182). happensAt(walking(lbgt_0),182). orientation(lbgt_1,152,182). coords(lbgt_1,79,98,182).  visible(lbgt_1,182). happensAt(walking(lbgt_1),182).
orientation(lbgt_0,168,183). coords(lbgt_0,209,192,183).  visible(lbgt_0,183). happensAt(walking(lbgt_0),183). orientation(lbgt_1,152,183). coords(lbgt_1,79,97,183).  visible(lbgt_1,183). happensAt(walking(lbgt_1),183).
orientation(lbgt_0,168,184). coords(lbgt_0,208,191,184).  visible(lbgt_0,184). happensAt(walking(lbgt_0),184). orientation(lbgt_1,152,184). coords(lbgt_1,79,97,184).  visible(lbgt_1,184). happensAt(walking(lbgt_1),184).
orientation(lbgt_0,168,185). coords(lbgt_0,208,189,185).  visible(lbgt_0,185). happensAt(walking(lbgt_0),185). orientation(lbgt_1,152,185). coords(lbgt_1,79,97,185).  visible(lbgt_1,185). happensAt(walking(lbgt_1),185).
orientation(lbgt_0,160,186). coords(lbgt_0,208,188,186).  visible(lbgt_0,186). happensAt(walking(lbgt_0),186). orientation(lbgt_1,152,186). coords(lbgt_1,79,97,186).  visible(lbgt_1,186). happensAt(walking(lbgt_1),186).
orientation(lbgt_0,160,187). coords(lbgt_0,207,186,187).  visible(lbgt_0,187). happensAt(walking(lbgt_0),187). orientation(lbgt_1,152,187). coords(lbgt_1,79,97,187).  visible(lbgt_1,187). happensAt(walking(lbgt_1),187).
orientation(lbgt_0,160,188). coords(lbgt_0,207,185,188).  visible(lbgt_0,188). happensAt(walking(lbgt_0),188). orientation(lbgt_1,152,188). coords(lbgt_1,79,97,188).  visible(lbgt_1,188). happensAt(walking(lbgt_1),188).
orientation(lbgt_0,160,189). coords(lbgt_0,207,185,189).  visible(lbgt_0,189). happensAt(walking(lbgt_0),189). orientation(lbgt_1,152,189). coords(lbgt_1,79,96,189).  visible(lbgt_1,189). happensAt(walking(lbgt_1),189).
orientation(lbgt_0,160,190). coords(lbgt_0,207,184,190).  visible(lbgt_0,190). happensAt(walking(lbgt_0),190). orientation(lbgt_1,152,190). coords(lbgt_1,79,95,190).  visible(lbgt_1,190). happensAt(walking(lbgt_1),190).
orientation(lbgt_0,160,191). coords(lbgt_0,206,183,191).  visible(lbgt_0,191). happensAt(walking(lbgt_0),191). orientation(lbgt_1,152,191). coords(lbgt_1,79,96,191).  visible(lbgt_1,191). happensAt(walking(lbgt_1),191).
orientation(lbgt_0,156,192). coords(lbgt_0,206,182,192).  visible(lbgt_0,192). happensAt(walking(lbgt_0),192). orientation(lbgt_1,152,192). coords(lbgt_1,79,95,192).  visible(lbgt_1,192). happensAt(walking(lbgt_1),192).
orientation(lbgt_0,153,193). coords(lbgt_0,204,182,193).  visible(lbgt_0,193). happensAt(walking(lbgt_0),193). orientation(lbgt_1,152,193). coords(lbgt_1,79,95,193).  visible(lbgt_1,193). happensAt(walking(lbgt_1),193).
orientation(lbgt_0,150,194). coords(lbgt_0,204,181,194).  visible(lbgt_0,194). happensAt(walking(lbgt_0),194). orientation(lbgt_1,152,194). coords(lbgt_1,80,94,194).  visible(lbgt_1,194). happensAt(walking(lbgt_1),194).
orientation(lbgt_0,150,195). coords(lbgt_0,203,180,195).  visible(lbgt_0,195). happensAt(walking(lbgt_0),195). orientation(lbgt_1,147,195). coords(lbgt_1,80,94,195).  visible(lbgt_1,195). happensAt(walking(lbgt_1),195).
orientation(lbgt_0,150,196). coords(lbgt_0,202,179,196).  visible(lbgt_0,196). happensAt(walking(lbgt_0),196). orientation(lbgt_1,147,196). coords(lbgt_1,80,92,196).  visible(lbgt_1,196). happensAt(walking(lbgt_1),196).
orientation(lbgt_0,150,197). coords(lbgt_0,200,178,197).  visible(lbgt_0,197). happensAt(walking(lbgt_0),197). orientation(lbgt_1,147,197). coords(lbgt_1,80,92,197).  visible(lbgt_1,197). happensAt(walking(lbgt_1),197).
orientation(lbgt_0,150,198). coords(lbgt_0,199,176,198).  visible(lbgt_0,198). happensAt(walking(lbgt_0),198). orientation(lbgt_1,147,198). coords(lbgt_1,80,92,198).  visible(lbgt_1,198). happensAt(walking(lbgt_1),198).
orientation(lbgt_0,150,199). coords(lbgt_0,198,174,199).  visible(lbgt_0,199). happensAt(walking(lbgt_0),199). orientation(lbgt_1,147,199). coords(lbgt_1,80,91,199).  visible(lbgt_1,199). happensAt(walking(lbgt_1),199).
orientation(lbgt_0,150,200). coords(lbgt_0,196,172,200).  visible(lbgt_0,200). happensAt(walking(lbgt_0),200). orientation(lbgt_1,147,200). coords(lbgt_1,80,90,200).  visible(lbgt_1,200). happensAt(walking(lbgt_1),200).
orientation(lbgt_0,150,201). coords(lbgt_0,194,169,201).  visible(lbgt_0,201). happensAt(walking(lbgt_0),201). orientation(lbgt_1,147,201). coords(lbgt_1,80,89,201).  visible(lbgt_1,201). happensAt(walking(lbgt_1),201).
orientation(lbgt_0,150,202). coords(lbgt_0,192,167,202).  visible(lbgt_0,202). happensAt(walking(lbgt_0),202). orientation(lbgt_1,147,202). coords(lbgt_1,80,89,202).  visible(lbgt_1,202). happensAt(walking(lbgt_1),202).
orientation(lbgt_0,150,203). coords(lbgt_0,190,166,203).  visible(lbgt_0,203). happensAt(walking(lbgt_0),203). orientation(lbgt_1,147,203). coords(lbgt_1,80,89,203).  visible(lbgt_1,203). happensAt(walking(lbgt_1),203).
orientation(lbgt_0,150,204). coords(lbgt_0,189,165,204).  visible(lbgt_0,204). happensAt(walking(lbgt_0),204). orientation(lbgt_1,147,204). coords(lbgt_1,80,88,204).  visible(lbgt_1,204). happensAt(walking(lbgt_1),204).
#show holdsAt/2.