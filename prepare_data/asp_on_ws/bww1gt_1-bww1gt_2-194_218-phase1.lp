
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
orientation(bww1gt_1,176,207). coords(bww1gt_1,109,270,207).  visible(bww1gt_1,207). happensAt(walking(bww1gt_1),207). orientation(bww1gt_2,167,207). coords(bww1gt_2,33,165,207).  visible(bww1gt_2,207). happensAt(walking(bww1gt_2),207).
orientation(bww1gt_1,177,208). coords(bww1gt_1,108,268,208).  visible(bww1gt_1,208). happensAt(walking(bww1gt_1),208). orientation(bww1gt_2,167,208). coords(bww1gt_2,33,165,208).  visible(bww1gt_2,208). happensAt(walking(bww1gt_2),208).
orientation(bww1gt_1,179,209). coords(bww1gt_1,107,267,209).  visible(bww1gt_1,209). happensAt(walking(bww1gt_1),209). orientation(bww1gt_2,167,209). coords(bww1gt_2,33,165,209).  visible(bww1gt_2,209). happensAt(walking(bww1gt_2),209).
orientation(bww1gt_1,178,210). coords(bww1gt_1,105,267,210).  visible(bww1gt_1,210). happensAt(walking(bww1gt_1),210). orientation(bww1gt_2,167,210). coords(bww1gt_2,33,165,210).  visible(bww1gt_2,210). happensAt(walking(bww1gt_2),210).
orientation(bww1gt_1,1,211). coords(bww1gt_1,104,266,211).  visible(bww1gt_1,211). happensAt(walking(bww1gt_1),211). orientation(bww1gt_2,167,211). coords(bww1gt_2,33,165,211).  visible(bww1gt_2,211). happensAt(walking(bww1gt_2),211).
orientation(bww1gt_1,177,212). coords(bww1gt_1,103,266,212).  visible(bww1gt_1,212). happensAt(walking(bww1gt_1),212). orientation(bww1gt_2,167,212). coords(bww1gt_2,34,165,212).  visible(bww1gt_2,212). happensAt(walking(bww1gt_2),212).
orientation(bww1gt_1,177,213). coords(bww1gt_1,103,265,213).  visible(bww1gt_1,213). happensAt(walking(bww1gt_1),213). orientation(bww1gt_2,167,213). coords(bww1gt_2,34,165,213).  visible(bww1gt_2,213). happensAt(walking(bww1gt_2),213).
orientation(bww1gt_1,177,214). coords(bww1gt_1,102,264,214).  visible(bww1gt_1,214). happensAt(walking(bww1gt_1),214). orientation(bww1gt_2,167,214). coords(bww1gt_2,34,165,214).  visible(bww1gt_2,214). happensAt(walking(bww1gt_2),214).
orientation(bww1gt_1,179,215). coords(bww1gt_1,101,264,215).  visible(bww1gt_1,215). happensAt(walking(bww1gt_1),215). orientation(bww1gt_2,167,215). coords(bww1gt_2,35,165,215).  visible(bww1gt_2,215). happensAt(walking(bww1gt_2),215).
orientation(bww1gt_1,179,216). coords(bww1gt_1,100,264,216).  visible(bww1gt_1,216). happensAt(walking(bww1gt_1),216). orientation(bww1gt_2,167,216). coords(bww1gt_2,35,165,216).  visible(bww1gt_2,216). happensAt(walking(bww1gt_2),216).
orientation(bww1gt_1,176,217). coords(bww1gt_1,100,264,217).  visible(bww1gt_1,217). happensAt(walking(bww1gt_1),217). orientation(bww1gt_2,167,217). coords(bww1gt_2,35,165,217).  visible(bww1gt_2,217). happensAt(walking(bww1gt_2),217).
orientation(bww1gt_1,175,218). coords(bww1gt_1,99,264,218).  visible(bww1gt_1,218). happensAt(walking(bww1gt_1),218). orientation(bww1gt_2,167,218). coords(bww1gt_2,35,165,218).  visible(bww1gt_2,218). happensAt(walking(bww1gt_2),218).
#show holdsAt/2.