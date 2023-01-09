
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

	orientation(br1gt_2,12,862). coords(br1gt_2,85,208,862).  visible(br1gt_2,862). happensAt(inactive(br1gt_2),862). orientation(br1gt_3,152,862). coords(br1gt_3,289,182,862).  visible(br1gt_3,862). happensAt(walking(br1gt_3),862).
orientation(br1gt_2,12,863). coords(br1gt_2,85,208,863).  visible(br1gt_2,863). happensAt(inactive(br1gt_2),863). orientation(br1gt_3,152,863). coords(br1gt_3,289,182,863).  visible(br1gt_3,863). happensAt(walking(br1gt_3),863).
orientation(br1gt_2,12,864). coords(br1gt_2,85,208,864).  visible(br1gt_2,864). happensAt(inactive(br1gt_2),864). orientation(br1gt_3,152,864). coords(br1gt_3,289,182,864).  visible(br1gt_3,864). happensAt(walking(br1gt_3),864).
orientation(br1gt_2,12,865). coords(br1gt_2,85,205,865).  visible(br1gt_2,865). happensAt(inactive(br1gt_2),865). orientation(br1gt_3,152,865). coords(br1gt_3,289,181,865).  visible(br1gt_3,865). happensAt(walking(br1gt_3),865).
orientation(br1gt_2,12,866). coords(br1gt_2,85,208,866).  visible(br1gt_2,866). happensAt(inactive(br1gt_2),866). orientation(br1gt_3,145,866). coords(br1gt_3,289,180,866).  visible(br1gt_3,866). happensAt(walking(br1gt_3),866).
orientation(br1gt_2,12,867). coords(br1gt_2,85,208,867).  visible(br1gt_2,867). happensAt(inactive(br1gt_2),867). orientation(br1gt_3,145,867). coords(br1gt_3,289,178,867).  visible(br1gt_3,867). happensAt(walking(br1gt_3),867).
orientation(br1gt_2,12,868). coords(br1gt_2,85,207,868).  visible(br1gt_2,868). happensAt(inactive(br1gt_2),868). orientation(br1gt_3,104,868). coords(br1gt_3,289,176,868).  visible(br1gt_3,868). happensAt(walking(br1gt_3),868).
orientation(br1gt_2,12,869). coords(br1gt_2,85,207,869).  visible(br1gt_2,869). happensAt(inactive(br1gt_2),869). orientation(br1gt_3,104,869). coords(br1gt_3,289,174,869).  visible(br1gt_3,869). happensAt(walking(br1gt_3),869).
orientation(br1gt_2,5,870). coords(br1gt_2,85,207,870).  visible(br1gt_2,870). happensAt(inactive(br1gt_2),870). orientation(br1gt_3,104,870). coords(br1gt_3,290,173,870).  visible(br1gt_3,870). happensAt(walking(br1gt_3),870).
orientation(br1gt_2,5,871). coords(br1gt_2,85,207,871).  visible(br1gt_2,871). happensAt(inactive(br1gt_2),871). orientation(br1gt_3,104,871). coords(br1gt_3,291,173,871).  visible(br1gt_3,871). happensAt(walking(br1gt_3),871).
orientation(br1gt_2,5,872). coords(br1gt_2,85,206,872).  visible(br1gt_2,872). happensAt(inactive(br1gt_2),872). orientation(br1gt_3,104,872). coords(br1gt_3,292,172,872).  visible(br1gt_3,872). happensAt(walking(br1gt_3),872).
orientation(br1gt_2,5,873). coords(br1gt_2,85,206,873).  visible(br1gt_2,873). happensAt(inactive(br1gt_2),873). orientation(br1gt_3,104,873). coords(br1gt_3,293,171,873).  visible(br1gt_3,873). happensAt(walking(br1gt_3),873).
orientation(br1gt_2,5,874). coords(br1gt_2,85,206,874).  visible(br1gt_2,874). happensAt(inactive(br1gt_2),874). orientation(br1gt_3,104,874). coords(br1gt_3,293,170,874).  visible(br1gt_3,874). happensAt(walking(br1gt_3),874).
orientation(br1gt_2,5,875). coords(br1gt_2,85,205,875).  visible(br1gt_2,875). happensAt(inactive(br1gt_2),875). orientation(br1gt_3,104,875). coords(br1gt_3,293,169,875).  visible(br1gt_3,875). happensAt(walking(br1gt_3),875).
orientation(br1gt_2,5,876). coords(br1gt_2,85,205,876).  visible(br1gt_2,876). happensAt(inactive(br1gt_2),876). orientation(br1gt_3,104,876). coords(br1gt_3,293,168,876).  visible(br1gt_3,876). happensAt(walking(br1gt_3),876).
orientation(br1gt_2,5,877). coords(br1gt_2,85,205,877).  visible(br1gt_2,877). happensAt(inactive(br1gt_2),877). orientation(br1gt_3,104,877). coords(br1gt_3,294,167,877).  visible(br1gt_3,877). happensAt(walking(br1gt_3),877).
orientation(br1gt_2,5,878). coords(br1gt_2,85,205,878).  visible(br1gt_2,878). happensAt(inactive(br1gt_2),878). orientation(br1gt_3,104,878). coords(br1gt_3,294,166,878).  visible(br1gt_3,878). happensAt(walking(br1gt_3),878).
orientation(br1gt_2,5,879). coords(br1gt_2,85,205,879).  visible(br1gt_2,879). happensAt(inactive(br1gt_2),879). orientation(br1gt_3,104,879). coords(br1gt_3,293,164,879).  visible(br1gt_3,879). happensAt(walking(br1gt_3),879).
orientation(br1gt_2,5,880). coords(br1gt_2,86,205,880).  visible(br1gt_2,880). happensAt(inactive(br1gt_2),880). orientation(br1gt_3,104,880). coords(br1gt_3,293,163,880).  visible(br1gt_3,880). happensAt(walking(br1gt_3),880).
orientation(br1gt_2,5,881). coords(br1gt_2,85,204,881).  visible(br1gt_2,881). happensAt(inactive(br1gt_2),881). orientation(br1gt_3,104,881). coords(br1gt_3,292,162,881).  visible(br1gt_3,881). happensAt(walking(br1gt_3),881).
orientation(br1gt_2,5,882). coords(br1gt_2,85,204,882).  visible(br1gt_2,882). happensAt(inactive(br1gt_2),882). orientation(br1gt_3,104,882). coords(br1gt_3,289,160,882).  visible(br1gt_3,882). happensAt(walking(br1gt_3),882).
orientation(br1gt_2,5,883). coords(br1gt_2,85,204,883).  visible(br1gt_2,883). happensAt(inactive(br1gt_2),883). orientation(br1gt_3,104,883). coords(br1gt_3,289,157,883).  visible(br1gt_3,883). happensAt(walking(br1gt_3),883).
orientation(br1gt_2,2,884). coords(br1gt_2,85,204,884).  visible(br1gt_2,884). happensAt(inactive(br1gt_2),884). orientation(br1gt_3,104,884). coords(br1gt_3,289,154,884).  visible(br1gt_3,884). happensAt(walking(br1gt_3),884).
orientation(br1gt_2,2,885). coords(br1gt_2,85,204,885).  visible(br1gt_2,885). happensAt(inactive(br1gt_2),885). orientation(br1gt_3,104,885). coords(br1gt_3,289,152,885).  visible(br1gt_3,885). happensAt(walking(br1gt_3),885).
orientation(br1gt_2,2,886). coords(br1gt_2,85,204,886).  visible(br1gt_2,886). happensAt(inactive(br1gt_2),886). orientation(br1gt_3,104,886). coords(br1gt_3,288,149,886).  visible(br1gt_3,886). happensAt(walking(br1gt_3),886).
#show holdsAt/2.