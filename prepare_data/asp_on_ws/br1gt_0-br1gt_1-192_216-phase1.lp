
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

	orientation(br1gt_0,47,192). coords(br1gt_0,301,278,192).  visible(br1gt_0,192). happensAt(walking(br1gt_0),192). orientation(br1gt_1,164,192). coords(br1gt_1,35,156,192).  visible(br1gt_1,192). happensAt(walking(br1gt_1),192).
orientation(br1gt_0,47,193). coords(br1gt_0,300,278,193).  visible(br1gt_0,193). happensAt(walking(br1gt_0),193). orientation(br1gt_1,164,193). coords(br1gt_1,35,156,193).  visible(br1gt_1,193). happensAt(walking(br1gt_1),193).
orientation(br1gt_0,47,194). coords(br1gt_0,304,278,194).  visible(br1gt_0,194). happensAt(walking(br1gt_0),194). orientation(br1gt_1,164,194). coords(br1gt_1,35,156,194).  visible(br1gt_1,194). happensAt(walking(br1gt_1),194).
orientation(br1gt_0,47,195). coords(br1gt_0,304,278,195).  visible(br1gt_0,195). happensAt(walking(br1gt_0),195). orientation(br1gt_1,164,195). coords(br1gt_1,34,156,195).  visible(br1gt_1,195). happensAt(walking(br1gt_1),195).
orientation(br1gt_0,47,196). coords(br1gt_0,305,279,196).  visible(br1gt_0,196). happensAt(walking(br1gt_0),196). orientation(br1gt_1,164,196). coords(br1gt_1,34,156,196).  visible(br1gt_1,196). happensAt(walking(br1gt_1),196).
orientation(br1gt_0,47,197). coords(br1gt_0,305,279,197).  visible(br1gt_0,197). happensAt(walking(br1gt_0),197). orientation(br1gt_1,164,197). coords(br1gt_1,33,155,197).  visible(br1gt_1,197). happensAt(walking(br1gt_1),197).
orientation(br1gt_0,47,198). coords(br1gt_0,304,279,198).  visible(br1gt_0,198). happensAt(walking(br1gt_0),198). orientation(br1gt_1,164,198). coords(br1gt_1,33,155,198).  visible(br1gt_1,198). happensAt(walking(br1gt_1),198).
orientation(br1gt_0,47,199). coords(br1gt_0,305,279,199).  visible(br1gt_0,199). happensAt(walking(br1gt_0),199). orientation(br1gt_1,164,199). coords(br1gt_1,32,155,199).  visible(br1gt_1,199). happensAt(walking(br1gt_1),199).
orientation(br1gt_0,47,200). coords(br1gt_0,305,280,200).  visible(br1gt_0,200). happensAt(walking(br1gt_0),200). orientation(br1gt_1,164,200). coords(br1gt_1,33,155,200).  visible(br1gt_1,200). happensAt(walking(br1gt_1),200).
orientation(br1gt_0,47,201). coords(br1gt_0,305,280,201).  visible(br1gt_0,201). happensAt(walking(br1gt_0),201). orientation(br1gt_1,164,201). coords(br1gt_1,32,157,201).  visible(br1gt_1,201). happensAt(walking(br1gt_1),201).
orientation(br1gt_0,47,202). coords(br1gt_0,305,280,202).  visible(br1gt_0,202). happensAt(walking(br1gt_0),202). orientation(br1gt_1,164,202). coords(br1gt_1,32,157,202).  visible(br1gt_1,202). happensAt(walking(br1gt_1),202).
orientation(br1gt_0,47,203). coords(br1gt_0,305,280,203).  visible(br1gt_0,203). happensAt(walking(br1gt_0),203). orientation(br1gt_1,164,203). coords(br1gt_1,32,156,203).  visible(br1gt_1,203). happensAt(walking(br1gt_1),203).
orientation(br1gt_0,47,204). coords(br1gt_0,305,280,204).  visible(br1gt_0,204). happensAt(walking(br1gt_0),204). orientation(br1gt_1,164,204). coords(br1gt_1,32,157,204).  visible(br1gt_1,204). happensAt(walking(br1gt_1),204).
orientation(br1gt_0,47,205). coords(br1gt_0,305,281,205).  visible(br1gt_0,205). happensAt(walking(br1gt_0),205). orientation(br1gt_1,164,205). coords(br1gt_1,31,157,205).  visible(br1gt_1,205). happensAt(walking(br1gt_1),205).
orientation(br1gt_0,47,206). coords(br1gt_0,305,281,206).  visible(br1gt_0,206). happensAt(walking(br1gt_0),206). orientation(br1gt_1,164,206). coords(br1gt_1,31,157,206).  visible(br1gt_1,206). happensAt(walking(br1gt_1),206).
orientation(br1gt_0,47,207). coords(br1gt_0,305,281,207).  visible(br1gt_0,207). happensAt(walking(br1gt_0),207). orientation(br1gt_1,164,207). coords(br1gt_1,31,157,207).  visible(br1gt_1,207). happensAt(walking(br1gt_1),207).
orientation(br1gt_0,47,208). coords(br1gt_0,305,281,208).  visible(br1gt_0,208). happensAt(walking(br1gt_0),208). orientation(br1gt_1,164,208). coords(br1gt_1,31,157,208).  visible(br1gt_1,208). happensAt(walking(br1gt_1),208).
orientation(br1gt_0,47,209). coords(br1gt_0,305,281,209).  visible(br1gt_0,209). happensAt(walking(br1gt_0),209). orientation(br1gt_1,162,209). coords(br1gt_1,31,158,209).  visible(br1gt_1,209). happensAt(walking(br1gt_1),209).
orientation(br1gt_0,47,210). coords(br1gt_0,305,281,210).  visible(br1gt_0,210). happensAt(walking(br1gt_0),210). orientation(br1gt_1,162,210). coords(br1gt_1,31,158,210).  visible(br1gt_1,210). happensAt(walking(br1gt_1),210).
orientation(br1gt_0,47,211). coords(br1gt_0,305,281,211).  visible(br1gt_0,211). happensAt(walking(br1gt_0),211). orientation(br1gt_1,170,211). coords(br1gt_1,31,158,211).  visible(br1gt_1,211). happensAt(walking(br1gt_1),211).
orientation(br1gt_0,47,212). coords(br1gt_0,305,282,212).  visible(br1gt_0,212). happensAt(walking(br1gt_0),212). orientation(br1gt_1,170,212). coords(br1gt_1,31,158,212).  visible(br1gt_1,212). happensAt(walking(br1gt_1),212).
orientation(br1gt_0,47,213). coords(br1gt_0,305,282,213).  visible(br1gt_0,213). happensAt(walking(br1gt_0),213). orientation(br1gt_1,170,213). coords(br1gt_1,31,158,213).  visible(br1gt_1,213). happensAt(walking(br1gt_1),213).
orientation(br1gt_0,47,214). coords(br1gt_0,305,282,214).  visible(br1gt_0,214). happensAt(walking(br1gt_0),214). orientation(br1gt_1,170,214). coords(br1gt_1,30,159,214).  visible(br1gt_1,214). happensAt(walking(br1gt_1),214).
orientation(br1gt_0,53,215). coords(br1gt_0,307,282,215).  visible(br1gt_0,215). happensAt(walking(br1gt_0),215). orientation(br1gt_1,170,215). coords(br1gt_1,30,159,215).  visible(br1gt_1,215). happensAt(walking(br1gt_1),215).
orientation(br1gt_0,53,216). coords(br1gt_0,307,282,216).  visible(br1gt_0,216). happensAt(walking(br1gt_0),216). orientation(br1gt_1,170,216). coords(br1gt_1,30,159,216).  visible(br1gt_1,216). happensAt(walking(br1gt_1),216).
#show holdsAt/2.