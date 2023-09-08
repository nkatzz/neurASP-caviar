
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
orientation(br2gt_1,78,207). coords(br2gt_1,285,262,207).  visible(br2gt_1,207). happensAt(walking(br2gt_1),207). orientation(br2gt_2,144,207). coords(br2gt_2,76,69,207).  visible(br2gt_2,207). happensAt(walking(br2gt_2),207).
orientation(br2gt_1,78,208). coords(br2gt_1,286,264,208).  visible(br2gt_1,208). happensAt(walking(br2gt_1),208). orientation(br2gt_2,144,208). coords(br2gt_2,76,69,208).  visible(br2gt_2,208). happensAt(walking(br2gt_2),208).
orientation(br2gt_1,71,209). coords(br2gt_1,288,267,209).  visible(br2gt_1,209). happensAt(walking(br2gt_1),209). orientation(br2gt_2,144,209). coords(br2gt_2,76,69,209).  visible(br2gt_2,209). happensAt(walking(br2gt_2),209).
orientation(br2gt_1,71,210). coords(br2gt_1,291,270,210).  visible(br2gt_1,210). happensAt(walking(br2gt_1),210). orientation(br2gt_2,144,210). coords(br2gt_2,76,70,210).  visible(br2gt_2,210). happensAt(walking(br2gt_2),210).
orientation(br2gt_1,71,211). coords(br2gt_1,293,273,211).  visible(br2gt_1,211). happensAt(walking(br2gt_1),211). orientation(br2gt_2,144,211). coords(br2gt_2,76,70,211).  visible(br2gt_2,211). happensAt(walking(br2gt_2),211).
orientation(br2gt_1,71,212). coords(br2gt_1,296,274,212).  visible(br2gt_1,212). happensAt(walking(br2gt_1),212). orientation(br2gt_2,144,212). coords(br2gt_2,76,70,212).  visible(br2gt_2,212). happensAt(walking(br2gt_2),212).
orientation(br2gt_1,71,213). coords(br2gt_1,297,274,213).  visible(br2gt_1,213). happensAt(walking(br2gt_1),213). orientation(br2gt_2,144,213). coords(br2gt_2,76,70,213).  visible(br2gt_2,213). happensAt(walking(br2gt_2),213).
orientation(br2gt_1,71,214). coords(br2gt_1,298,274,214).  visible(br2gt_1,214). happensAt(walking(br2gt_1),214). orientation(br2gt_2,144,214). coords(br2gt_2,76,70,214).  visible(br2gt_2,214). happensAt(walking(br2gt_2),214).
orientation(br2gt_1,71,215). coords(br2gt_1,299,274,215).  visible(br2gt_1,215). happensAt(walking(br2gt_1),215). orientation(br2gt_2,144,215). coords(br2gt_2,76,70,215).  visible(br2gt_2,215). happensAt(walking(br2gt_2),215).
orientation(br2gt_1,71,216). coords(br2gt_1,299,274,216).  visible(br2gt_1,216). happensAt(walking(br2gt_1),216). orientation(br2gt_2,144,216). coords(br2gt_2,76,70,216).  visible(br2gt_2,216). happensAt(walking(br2gt_2),216).
orientation(br2gt_1,71,217). coords(br2gt_1,298,274,217).  visible(br2gt_1,217). happensAt(walking(br2gt_1),217). orientation(br2gt_2,144,217). coords(br2gt_2,76,70,217).  visible(br2gt_2,217). happensAt(walking(br2gt_2),217).
orientation(br2gt_1,71,218). coords(br2gt_1,299,275,218).  visible(br2gt_1,218). happensAt(walking(br2gt_1),218). orientation(br2gt_2,144,218). coords(br2gt_2,76,70,218).  visible(br2gt_2,218). happensAt(walking(br2gt_2),218).
#show holdsAt/2.