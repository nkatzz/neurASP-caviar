
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

	orientation(br1gt_2,2,910). coords(br1gt_2,85,200,910).  visible(br1gt_2,910). happensAt(active(br1gt_2),910). orientation(br1gt_3,89,910). coords(br1gt_3,292,111,910).  visible(br1gt_3,910). happensAt(walking(br1gt_3),910).
orientation(br1gt_2,2,911). coords(br1gt_2,85,200,911).  visible(br1gt_2,911). happensAt(active(br1gt_2),911). orientation(br1gt_3,89,911). coords(br1gt_3,294,110,911).  visible(br1gt_3,911). happensAt(walking(br1gt_3),911).
orientation(br1gt_2,2,912). coords(br1gt_2,85,200,912).  visible(br1gt_2,912). happensAt(active(br1gt_2),912). orientation(br1gt_3,89,912). coords(br1gt_3,294,110,912).  visible(br1gt_3,912). happensAt(walking(br1gt_3),912).
orientation(br1gt_2,2,913). coords(br1gt_2,85,199,913).  visible(br1gt_2,913). happensAt(active(br1gt_2),913). orientation(br1gt_3,89,913). coords(br1gt_3,294,109,913).  visible(br1gt_3,913). happensAt(walking(br1gt_3),913).
orientation(br1gt_2,2,914). coords(br1gt_2,85,199,914).  visible(br1gt_2,914). happensAt(active(br1gt_2),914). orientation(br1gt_3,89,914). coords(br1gt_3,295,109,914).  visible(br1gt_3,914). happensAt(walking(br1gt_3),914).
orientation(br1gt_2,2,915). coords(br1gt_2,85,199,915).  visible(br1gt_2,915). happensAt(active(br1gt_2),915). orientation(br1gt_3,89,915). coords(br1gt_3,296,108,915).  visible(br1gt_3,915). happensAt(walking(br1gt_3),915).
orientation(br1gt_2,2,916). coords(br1gt_2,85,199,916).  visible(br1gt_2,916). happensAt(active(br1gt_2),916). orientation(br1gt_3,89,916). coords(br1gt_3,295,107,916).  visible(br1gt_3,916). happensAt(walking(br1gt_3),916).
orientation(br1gt_2,2,917). coords(br1gt_2,85,199,917).  visible(br1gt_2,917). happensAt(active(br1gt_2),917). orientation(br1gt_3,89,917). coords(br1gt_3,295,106,917).  visible(br1gt_3,917). happensAt(walking(br1gt_3),917).
orientation(br1gt_2,2,918). coords(br1gt_2,85,199,918).  visible(br1gt_2,918). happensAt(active(br1gt_2),918). orientation(br1gt_3,89,918). coords(br1gt_3,295,105,918).  visible(br1gt_3,918). happensAt(walking(br1gt_3),918).
orientation(br1gt_2,2,919). coords(br1gt_2,85,199,919).  visible(br1gt_2,919). happensAt(active(br1gt_2),919). orientation(br1gt_3,89,919). coords(br1gt_3,295,103,919).  visible(br1gt_3,919). happensAt(walking(br1gt_3),919).
orientation(br1gt_2,2,920). coords(br1gt_2,85,199,920).  visible(br1gt_2,920). happensAt(active(br1gt_2),920). orientation(br1gt_3,89,920). coords(br1gt_3,296,101,920).  visible(br1gt_3,920). happensAt(walking(br1gt_3),920).
orientation(br1gt_2,2,921). coords(br1gt_2,86,197,921).  visible(br1gt_2,921). happensAt(active(br1gt_2),921). orientation(br1gt_3,89,921). coords(br1gt_3,297,99,921).  visible(br1gt_3,921). happensAt(walking(br1gt_3),921).
orientation(br1gt_2,2,922). coords(br1gt_2,86,198,922).  visible(br1gt_2,922). happensAt(active(br1gt_2),922). orientation(br1gt_3,89,922). coords(br1gt_3,297,98,922).  visible(br1gt_3,922). happensAt(active(br1gt_3),922).
orientation(br1gt_2,177,923). coords(br1gt_2,86,198,923).  visible(br1gt_2,923). happensAt(active(br1gt_2),923). orientation(br1gt_3,89,923). coords(br1gt_3,298,96,923).  visible(br1gt_3,923). happensAt(active(br1gt_3),923).
orientation(br1gt_2,177,924). coords(br1gt_2,86,198,924).  visible(br1gt_2,924). happensAt(active(br1gt_2),924). orientation(br1gt_3,89,924). coords(br1gt_3,299,96,924).  visible(br1gt_3,924). happensAt(active(br1gt_3),924).
orientation(br1gt_2,177,925). coords(br1gt_2,86,198,925).  visible(br1gt_2,925). happensAt(active(br1gt_2),925). orientation(br1gt_3,89,925). coords(br1gt_3,299,96,925).  visible(br1gt_3,925). happensAt(active(br1gt_3),925).
orientation(br1gt_2,174,926). coords(br1gt_2,86,197,926).  visible(br1gt_2,926). happensAt(active(br1gt_2),926). orientation(br1gt_3,89,926). coords(br1gt_3,299,95,926).  visible(br1gt_3,926). happensAt(active(br1gt_3),926).
orientation(br1gt_2,174,927). coords(br1gt_2,86,196,927).  visible(br1gt_2,927). happensAt(active(br1gt_2),927). orientation(br1gt_3,89,927). coords(br1gt_3,299,95,927).  visible(br1gt_3,927). happensAt(active(br1gt_3),927).
orientation(br1gt_2,171,928). coords(br1gt_2,86,198,928).  visible(br1gt_2,928). happensAt(active(br1gt_2),928). orientation(br1gt_3,89,928). coords(br1gt_3,299,95,928).  visible(br1gt_3,928). happensAt(active(br1gt_3),928).
orientation(br1gt_2,171,929). coords(br1gt_2,86,198,929).  visible(br1gt_2,929). happensAt(active(br1gt_2),929). orientation(br1gt_3,89,929). coords(br1gt_3,300,95,929).  visible(br1gt_3,929). happensAt(active(br1gt_3),929).
orientation(br1gt_2,171,930). coords(br1gt_2,86,198,930).  visible(br1gt_2,930). happensAt(walking(br1gt_2),930). orientation(br1gt_3,89,930). coords(br1gt_3,300,95,930).  visible(br1gt_3,930). happensAt(active(br1gt_3),930).
orientation(br1gt_2,171,931). coords(br1gt_2,86,198,931).  visible(br1gt_2,931). happensAt(walking(br1gt_2),931). orientation(br1gt_3,89,931). coords(br1gt_3,301,95,931).  visible(br1gt_3,931). happensAt(active(br1gt_3),931).
orientation(br1gt_2,171,932). coords(br1gt_2,86,198,932).  visible(br1gt_2,932). happensAt(walking(br1gt_2),932). orientation(br1gt_3,89,932). coords(br1gt_3,301,95,932).  visible(br1gt_3,932). happensAt(active(br1gt_3),932).
orientation(br1gt_2,171,933). coords(br1gt_2,86,198,933).  visible(br1gt_2,933). happensAt(walking(br1gt_2),933). orientation(br1gt_3,89,933). coords(br1gt_3,301,95,933).  visible(br1gt_3,933). happensAt(active(br1gt_3),933).
orientation(br1gt_2,171,934). coords(br1gt_2,86,199,934).  visible(br1gt_2,934). happensAt(walking(br1gt_2),934). orientation(br1gt_3,89,934). coords(br1gt_3,301,94,934).  visible(br1gt_3,934). happensAt(active(br1gt_3),934).
#show holdsAt/2.