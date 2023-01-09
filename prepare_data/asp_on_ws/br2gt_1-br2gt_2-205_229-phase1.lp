
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
orientation(br2gt_1,71,219). coords(br2gt_1,298,277,219).  visible(br2gt_1,219). happensAt(walking(br2gt_1),219). orientation(br2gt_2,144,219). coords(br2gt_2,76,70,219).  visible(br2gt_2,219). happensAt(walking(br2gt_2),219).
orientation(br2gt_1,71,220). coords(br2gt_1,305,279,220).  visible(br2gt_1,220). happensAt(walking(br2gt_1),220). orientation(br2gt_2,144,220). coords(br2gt_2,76,70,220).  visible(br2gt_2,220). happensAt(walking(br2gt_2),220).
orientation(br2gt_1,71,221). coords(br2gt_1,306,282,221).  visible(br2gt_1,221). happensAt(walking(br2gt_1),221). orientation(br2gt_2,144,221). coords(br2gt_2,76,70,221).  visible(br2gt_2,221). happensAt(walking(br2gt_2),221).
orientation(br2gt_1,71,222). coords(br2gt_1,306,284,222).  visible(br2gt_1,222). happensAt(walking(br2gt_1),222). orientation(br2gt_2,144,222). coords(br2gt_2,76,70,222).  visible(br2gt_2,222). happensAt(walking(br2gt_2),222).
orientation(br2gt_1,71,223). coords(br2gt_1,307,285,223).  visible(br2gt_1,223). happensAt(walking(br2gt_1),223). orientation(br2gt_2,144,223). coords(br2gt_2,76,70,223).  visible(br2gt_2,223). happensAt(walking(br2gt_2),223).
orientation(br2gt_1,71,224). coords(br2gt_1,307,285,224).  visible(br2gt_1,224). happensAt(walking(br2gt_1),224). orientation(br2gt_2,144,224). coords(br2gt_2,76,70,224).  visible(br2gt_2,224). happensAt(walking(br2gt_2),224).
orientation(br2gt_1,71,225). coords(br2gt_1,312,285,225).  visible(br2gt_1,225). happensAt(walking(br2gt_1),225). orientation(br2gt_2,144,225). coords(br2gt_2,76,70,225).  visible(br2gt_2,225). happensAt(walking(br2gt_2),225).
orientation(br2gt_1,71,226). coords(br2gt_1,312,285,226).  visible(br2gt_1,226). happensAt(walking(br2gt_1),226). orientation(br2gt_2,144,226). coords(br2gt_2,76,70,226).  visible(br2gt_2,226). happensAt(walking(br2gt_2),226).
orientation(br2gt_1,71,227). coords(br2gt_1,312,285,227).  visible(br2gt_1,227). happensAt(walking(br2gt_1),227). orientation(br2gt_2,144,227). coords(br2gt_2,76,70,227).  visible(br2gt_2,227). happensAt(walking(br2gt_2),227).
orientation(br2gt_1,71,228). coords(br2gt_1,313,286,228).  visible(br2gt_1,228). happensAt(walking(br2gt_1),228). orientation(br2gt_2,144,228). coords(br2gt_2,76,70,228).  visible(br2gt_2,228). happensAt(walking(br2gt_2),228).
orientation(br2gt_1,71,229). coords(br2gt_1,313,286,229).  happensAt(disappear(br2gt_1),229). happensAt(walking(br2gt_1),229). orientation(br2gt_2,144,229). coords(br2gt_2,76,70,229).  visible(br2gt_2,229). happensAt(walking(br2gt_2),229).
#show holdsAt/2.