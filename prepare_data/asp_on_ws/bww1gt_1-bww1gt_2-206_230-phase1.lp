
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
orientation(bww1gt_1,176,219). coords(bww1gt_1,98,264,219).  visible(bww1gt_1,219). happensAt(walking(bww1gt_1),219). orientation(bww1gt_2,167,219). coords(bww1gt_2,35,165,219).  visible(bww1gt_2,219). happensAt(walking(bww1gt_2),219).
orientation(bww1gt_1,176,220). coords(bww1gt_1,98,263,220).  visible(bww1gt_1,220). happensAt(walking(bww1gt_1),220). orientation(bww1gt_2,167,220). coords(bww1gt_2,35,165,220).  visible(bww1gt_2,220). happensAt(walking(bww1gt_2),220).
orientation(bww1gt_1,174,221). coords(bww1gt_1,97,263,221).  visible(bww1gt_1,221). happensAt(walking(bww1gt_1),221). orientation(bww1gt_2,167,221). coords(bww1gt_2,35,166,221).  visible(bww1gt_2,221). happensAt(walking(bww1gt_2),221).
orientation(bww1gt_1,173,222). coords(bww1gt_1,96,263,222).  visible(bww1gt_1,222). happensAt(walking(bww1gt_1),222). orientation(bww1gt_2,167,222). coords(bww1gt_2,35,166,222).  visible(bww1gt_2,222). happensAt(walking(bww1gt_2),222).
orientation(bww1gt_1,169,223). coords(bww1gt_1,94,263,223).  visible(bww1gt_1,223). happensAt(walking(bww1gt_1),223). orientation(bww1gt_2,167,223). coords(bww1gt_2,35,166,223).  visible(bww1gt_2,223). happensAt(walking(bww1gt_2),223).
orientation(bww1gt_1,173,224). coords(bww1gt_1,94,263,224).  visible(bww1gt_1,224). happensAt(walking(bww1gt_1),224). orientation(bww1gt_2,167,224). coords(bww1gt_2,35,166,224).  visible(bww1gt_2,224). happensAt(walking(bww1gt_2),224).
orientation(bww1gt_1,172,225). coords(bww1gt_1,94,262,225).  visible(bww1gt_1,225). happensAt(walking(bww1gt_1),225). orientation(bww1gt_2,167,225). coords(bww1gt_2,35,166,225).  visible(bww1gt_2,225). happensAt(walking(bww1gt_2),225).
orientation(bww1gt_1,174,226). coords(bww1gt_1,93,262,226).  visible(bww1gt_1,226). happensAt(walking(bww1gt_1),226). orientation(bww1gt_2,167,226). coords(bww1gt_2,37,166,226).  visible(bww1gt_2,226). happensAt(walking(bww1gt_2),226).
orientation(bww1gt_1,173,227). coords(bww1gt_1,92,260,227).  visible(bww1gt_1,227). happensAt(walking(bww1gt_1),227). orientation(bww1gt_2,167,227). coords(bww1gt_2,37,166,227).  visible(bww1gt_2,227). happensAt(walking(bww1gt_2),227).
orientation(bww1gt_1,178,228). coords(bww1gt_1,91,260,228).  visible(bww1gt_1,228). happensAt(walking(bww1gt_1),228). orientation(bww1gt_2,167,228). coords(bww1gt_2,38,166,228).  visible(bww1gt_2,228). happensAt(walking(bww1gt_2),228).
orientation(bww1gt_1,174,229). coords(bww1gt_1,89,259,229).  visible(bww1gt_1,229). happensAt(walking(bww1gt_1),229). orientation(bww1gt_2,167,229). coords(bww1gt_2,38,166,229).  visible(bww1gt_2,229). happensAt(walking(bww1gt_2),229).
orientation(bww1gt_1,170,230). coords(bww1gt_1,88,259,230).  visible(bww1gt_1,230). happensAt(walking(bww1gt_1),230). orientation(bww1gt_2,167,230). coords(bww1gt_2,38,166,230).  visible(bww1gt_2,230). happensAt(walking(bww1gt_2),230).
#show holdsAt/2.