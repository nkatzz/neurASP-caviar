
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

	orientation(br1gt_0,47,192). coords(br1gt_0,301,278,192).  visible(br1gt_0,192). happensAt(walking(br1gt_0),192). orientation(br1gt_2,171,192). coords(br1gt_2,86,207,192).  visible(br1gt_2,192). happensAt(inactive(br1gt_2),192).
orientation(br1gt_0,47,193). coords(br1gt_0,300,278,193).  visible(br1gt_0,193). happensAt(walking(br1gt_0),193). orientation(br1gt_2,171,193). coords(br1gt_2,86,207,193).  visible(br1gt_2,193). happensAt(inactive(br1gt_2),193).
orientation(br1gt_0,47,194). coords(br1gt_0,304,278,194).  visible(br1gt_0,194). happensAt(walking(br1gt_0),194). orientation(br1gt_2,171,194). coords(br1gt_2,86,207,194).  visible(br1gt_2,194). happensAt(inactive(br1gt_2),194).
orientation(br1gt_0,47,195). coords(br1gt_0,304,278,195).  visible(br1gt_0,195). happensAt(walking(br1gt_0),195). orientation(br1gt_2,171,195). coords(br1gt_2,86,207,195).  visible(br1gt_2,195). happensAt(inactive(br1gt_2),195).
orientation(br1gt_0,47,196). coords(br1gt_0,305,279,196).  visible(br1gt_0,196). happensAt(walking(br1gt_0),196). orientation(br1gt_2,171,196). coords(br1gt_2,86,207,196).  visible(br1gt_2,196). happensAt(inactive(br1gt_2),196).
orientation(br1gt_0,47,197). coords(br1gt_0,305,279,197).  visible(br1gt_0,197). happensAt(walking(br1gt_0),197). orientation(br1gt_2,171,197). coords(br1gt_2,86,207,197).  visible(br1gt_2,197). happensAt(inactive(br1gt_2),197).
orientation(br1gt_0,47,198). coords(br1gt_0,304,279,198).  visible(br1gt_0,198). happensAt(walking(br1gt_0),198). orientation(br1gt_2,171,198). coords(br1gt_2,86,207,198).  visible(br1gt_2,198). happensAt(inactive(br1gt_2),198).
orientation(br1gt_0,47,199). coords(br1gt_0,305,279,199).  visible(br1gt_0,199). happensAt(walking(br1gt_0),199). orientation(br1gt_2,171,199). coords(br1gt_2,86,207,199).  visible(br1gt_2,199). happensAt(inactive(br1gt_2),199).
orientation(br1gt_0,47,200). coords(br1gt_0,305,280,200).  visible(br1gt_0,200). happensAt(walking(br1gt_0),200). orientation(br1gt_2,171,200). coords(br1gt_2,86,207,200).  visible(br1gt_2,200). happensAt(inactive(br1gt_2),200).
orientation(br1gt_0,47,201). coords(br1gt_0,305,280,201).  visible(br1gt_0,201). happensAt(walking(br1gt_0),201). orientation(br1gt_2,171,201). coords(br1gt_2,86,207,201).  visible(br1gt_2,201). happensAt(inactive(br1gt_2),201).
orientation(br1gt_0,47,202). coords(br1gt_0,305,280,202).  visible(br1gt_0,202). happensAt(walking(br1gt_0),202). orientation(br1gt_2,171,202). coords(br1gt_2,86,207,202).  visible(br1gt_2,202). happensAt(inactive(br1gt_2),202).
orientation(br1gt_0,47,203). coords(br1gt_0,305,280,203).  visible(br1gt_0,203). happensAt(walking(br1gt_0),203). orientation(br1gt_2,171,203). coords(br1gt_2,86,207,203).  visible(br1gt_2,203). happensAt(inactive(br1gt_2),203).
orientation(br1gt_0,47,204). coords(br1gt_0,305,280,204).  visible(br1gt_0,204). happensAt(walking(br1gt_0),204). orientation(br1gt_2,171,204). coords(br1gt_2,86,207,204).  visible(br1gt_2,204). happensAt(inactive(br1gt_2),204).
orientation(br1gt_0,47,205). coords(br1gt_0,305,281,205).  visible(br1gt_0,205). happensAt(walking(br1gt_0),205). orientation(br1gt_2,171,205). coords(br1gt_2,86,207,205).  visible(br1gt_2,205). happensAt(inactive(br1gt_2),205).
orientation(br1gt_0,47,206). coords(br1gt_0,305,281,206).  visible(br1gt_0,206). happensAt(walking(br1gt_0),206). orientation(br1gt_2,171,206). coords(br1gt_2,86,207,206).  visible(br1gt_2,206). happensAt(inactive(br1gt_2),206).
orientation(br1gt_0,47,207). coords(br1gt_0,305,281,207).  visible(br1gt_0,207). happensAt(walking(br1gt_0),207). orientation(br1gt_2,171,207). coords(br1gt_2,86,207,207).  visible(br1gt_2,207). happensAt(inactive(br1gt_2),207).
orientation(br1gt_0,47,208). coords(br1gt_0,305,281,208).  visible(br1gt_0,208). happensAt(walking(br1gt_0),208). orientation(br1gt_2,171,208). coords(br1gt_2,86,207,208).  visible(br1gt_2,208). happensAt(inactive(br1gt_2),208).
orientation(br1gt_0,47,209). coords(br1gt_0,305,281,209).  visible(br1gt_0,209). happensAt(walking(br1gt_0),209). orientation(br1gt_2,171,209). coords(br1gt_2,86,207,209).  visible(br1gt_2,209). happensAt(inactive(br1gt_2),209).
orientation(br1gt_0,47,210). coords(br1gt_0,305,281,210).  visible(br1gt_0,210). happensAt(walking(br1gt_0),210). orientation(br1gt_2,171,210). coords(br1gt_2,86,207,210).  visible(br1gt_2,210). happensAt(inactive(br1gt_2),210).
orientation(br1gt_0,47,211). coords(br1gt_0,305,281,211).  visible(br1gt_0,211). happensAt(walking(br1gt_0),211). orientation(br1gt_2,171,211). coords(br1gt_2,86,207,211).  visible(br1gt_2,211). happensAt(inactive(br1gt_2),211).
orientation(br1gt_0,47,212). coords(br1gt_0,305,282,212).  visible(br1gt_0,212). happensAt(walking(br1gt_0),212). orientation(br1gt_2,171,212). coords(br1gt_2,86,207,212).  visible(br1gt_2,212). happensAt(inactive(br1gt_2),212).
orientation(br1gt_0,47,213). coords(br1gt_0,305,282,213).  visible(br1gt_0,213). happensAt(walking(br1gt_0),213). orientation(br1gt_2,171,213). coords(br1gt_2,86,207,213).  visible(br1gt_2,213). happensAt(inactive(br1gt_2),213).
orientation(br1gt_0,47,214). coords(br1gt_0,305,282,214).  visible(br1gt_0,214). happensAt(walking(br1gt_0),214). orientation(br1gt_2,171,214). coords(br1gt_2,86,207,214).  visible(br1gt_2,214). happensAt(inactive(br1gt_2),214).
orientation(br1gt_0,53,215). coords(br1gt_0,307,282,215).  visible(br1gt_0,215). happensAt(walking(br1gt_0),215). orientation(br1gt_2,171,215). coords(br1gt_2,86,207,215).  visible(br1gt_2,215). happensAt(inactive(br1gt_2),215).
orientation(br1gt_0,53,216). coords(br1gt_0,307,282,216).  visible(br1gt_0,216). happensAt(walking(br1gt_0),216). orientation(br1gt_2,171,216). coords(br1gt_2,86,207,216).  visible(br1gt_2,216). happensAt(inactive(br1gt_2),216).
#show holdsAt/2.