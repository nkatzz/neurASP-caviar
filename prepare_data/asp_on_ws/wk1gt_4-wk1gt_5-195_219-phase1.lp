
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

	orientation(wk1gt_4,166,195). coords(wk1gt_4,47,181,195).  visible(wk1gt_4,195). happensAt(inactive(wk1gt_4),195). orientation(wk1gt_5,162,195). coords(wk1gt_5,49,166,195).  visible(wk1gt_5,195). happensAt(inactive(wk1gt_5),195).
orientation(wk1gt_4,165,196). coords(wk1gt_4,47,181,196).  visible(wk1gt_4,196). happensAt(inactive(wk1gt_4),196). orientation(wk1gt_5,162,196). coords(wk1gt_5,49,166,196).  visible(wk1gt_5,196). happensAt(inactive(wk1gt_5),196).
orientation(wk1gt_4,169,197). coords(wk1gt_4,47,181,197).  visible(wk1gt_4,197). happensAt(inactive(wk1gt_4),197). orientation(wk1gt_5,162,197). coords(wk1gt_5,49,166,197).  visible(wk1gt_5,197). happensAt(inactive(wk1gt_5),197).
orientation(wk1gt_4,166,198). coords(wk1gt_4,47,181,198).  visible(wk1gt_4,198). happensAt(inactive(wk1gt_4),198). orientation(wk1gt_5,162,198). coords(wk1gt_5,50,166,198).  visible(wk1gt_5,198). happensAt(inactive(wk1gt_5),198).
orientation(wk1gt_4,165,199). coords(wk1gt_4,47,181,199).  visible(wk1gt_4,199). happensAt(inactive(wk1gt_4),199). orientation(wk1gt_5,162,199). coords(wk1gt_5,50,166,199).  visible(wk1gt_5,199). happensAt(inactive(wk1gt_5),199).
orientation(wk1gt_4,165,200). coords(wk1gt_4,47,181,200).  visible(wk1gt_4,200). happensAt(inactive(wk1gt_4),200). orientation(wk1gt_5,162,200). coords(wk1gt_5,50,166,200).  visible(wk1gt_5,200). happensAt(inactive(wk1gt_5),200).
orientation(wk1gt_4,164,201). coords(wk1gt_4,47,181,201).  visible(wk1gt_4,201). happensAt(inactive(wk1gt_4),201). orientation(wk1gt_5,162,201). coords(wk1gt_5,50,166,201).  visible(wk1gt_5,201). happensAt(inactive(wk1gt_5),201).
orientation(wk1gt_4,169,202). coords(wk1gt_4,47,181,202).  visible(wk1gt_4,202). happensAt(inactive(wk1gt_4),202). orientation(wk1gt_5,162,202). coords(wk1gt_5,50,166,202).  visible(wk1gt_5,202). happensAt(inactive(wk1gt_5),202).
orientation(wk1gt_4,166,203). coords(wk1gt_4,47,181,203).  visible(wk1gt_4,203). happensAt(inactive(wk1gt_4),203). orientation(wk1gt_5,162,203). coords(wk1gt_5,50,166,203).  visible(wk1gt_5,203). happensAt(inactive(wk1gt_5),203).
orientation(wk1gt_4,163,204). coords(wk1gt_4,47,181,204).  visible(wk1gt_4,204). happensAt(inactive(wk1gt_4),204). orientation(wk1gt_5,162,204). coords(wk1gt_5,50,166,204).  visible(wk1gt_5,204). happensAt(inactive(wk1gt_5),204).
orientation(wk1gt_4,168,205). coords(wk1gt_4,47,181,205).  visible(wk1gt_4,205). happensAt(inactive(wk1gt_4),205). orientation(wk1gt_5,162,205). coords(wk1gt_5,50,166,205).  visible(wk1gt_5,205). happensAt(inactive(wk1gt_5),205).
orientation(wk1gt_4,167,206). coords(wk1gt_4,47,181,206).  visible(wk1gt_4,206). happensAt(inactive(wk1gt_4),206). orientation(wk1gt_5,162,206). coords(wk1gt_5,50,166,206).  visible(wk1gt_5,206). happensAt(inactive(wk1gt_5),206).
orientation(wk1gt_4,168,207). coords(wk1gt_4,47,181,207).  visible(wk1gt_4,207). happensAt(inactive(wk1gt_4),207). orientation(wk1gt_5,162,207). coords(wk1gt_5,50,166,207).  visible(wk1gt_5,207). happensAt(inactive(wk1gt_5),207).
orientation(wk1gt_4,164,208). coords(wk1gt_4,47,181,208).  visible(wk1gt_4,208). happensAt(inactive(wk1gt_4),208). orientation(wk1gt_5,162,208). coords(wk1gt_5,50,166,208).  visible(wk1gt_5,208). happensAt(inactive(wk1gt_5),208).
orientation(wk1gt_4,164,209). coords(wk1gt_4,47,181,209).  visible(wk1gt_4,209). happensAt(inactive(wk1gt_4),209). orientation(wk1gt_5,162,209). coords(wk1gt_5,50,166,209).  visible(wk1gt_5,209). happensAt(inactive(wk1gt_5),209).
orientation(wk1gt_4,164,210). coords(wk1gt_4,47,181,210).  visible(wk1gt_4,210). happensAt(inactive(wk1gt_4),210). orientation(wk1gt_5,162,210). coords(wk1gt_5,50,166,210).  visible(wk1gt_5,210). happensAt(inactive(wk1gt_5),210).
orientation(wk1gt_4,163,211). coords(wk1gt_4,47,181,211).  visible(wk1gt_4,211). happensAt(inactive(wk1gt_4),211). orientation(wk1gt_5,162,211). coords(wk1gt_5,50,166,211).  visible(wk1gt_5,211). happensAt(inactive(wk1gt_5),211).
orientation(wk1gt_4,164,212). coords(wk1gt_4,47,181,212).  visible(wk1gt_4,212). happensAt(inactive(wk1gt_4),212). orientation(wk1gt_5,162,212). coords(wk1gt_5,50,166,212).  visible(wk1gt_5,212). happensAt(inactive(wk1gt_5),212).
orientation(wk1gt_4,165,213). coords(wk1gt_4,47,181,213).  visible(wk1gt_4,213). happensAt(inactive(wk1gt_4),213). orientation(wk1gt_5,162,213). coords(wk1gt_5,50,166,213).  visible(wk1gt_5,213). happensAt(inactive(wk1gt_5),213).
orientation(wk1gt_4,163,214). coords(wk1gt_4,47,181,214).  visible(wk1gt_4,214). happensAt(inactive(wk1gt_4),214). orientation(wk1gt_5,162,214). coords(wk1gt_5,50,166,214).  visible(wk1gt_5,214). happensAt(inactive(wk1gt_5),214).
orientation(wk1gt_4,163,215). coords(wk1gt_4,47,181,215).  visible(wk1gt_4,215). happensAt(inactive(wk1gt_4),215). orientation(wk1gt_5,162,215). coords(wk1gt_5,50,166,215).  visible(wk1gt_5,215). happensAt(inactive(wk1gt_5),215).
orientation(wk1gt_4,165,216). coords(wk1gt_4,47,181,216).  visible(wk1gt_4,216). happensAt(inactive(wk1gt_4),216). orientation(wk1gt_5,162,216). coords(wk1gt_5,50,166,216).  visible(wk1gt_5,216). happensAt(inactive(wk1gt_5),216).
orientation(wk1gt_4,164,217). coords(wk1gt_4,47,181,217).  visible(wk1gt_4,217). happensAt(inactive(wk1gt_4),217). orientation(wk1gt_5,162,217). coords(wk1gt_5,50,166,217).  visible(wk1gt_5,217). happensAt(inactive(wk1gt_5),217).
orientation(wk1gt_4,166,218). coords(wk1gt_4,47,181,218).  visible(wk1gt_4,218). happensAt(inactive(wk1gt_4),218). orientation(wk1gt_5,162,218). coords(wk1gt_5,50,166,218).  visible(wk1gt_5,218). happensAt(inactive(wk1gt_5),218).
orientation(wk1gt_4,164,219). coords(wk1gt_4,47,181,219).  visible(wk1gt_4,219). happensAt(inactive(wk1gt_4),219). orientation(wk1gt_5,162,219). coords(wk1gt_5,50,166,219).  visible(wk1gt_5,219). happensAt(inactive(wk1gt_5),219).
#show holdsAt/2.