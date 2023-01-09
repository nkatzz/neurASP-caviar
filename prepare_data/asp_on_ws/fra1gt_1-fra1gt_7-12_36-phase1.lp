
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

	orientation(fra1gt_1,112,12). coords(fra1gt_1,277,20,12).  visible(fra1gt_1,12). happensAt(walking(fra1gt_1),12). orientation(fra1gt_7,136,12). coords(fra1gt_7,141,42,12).  visible(fra1gt_7,12). happensAt(walking(fra1gt_7),12).
orientation(fra1gt_1,112,13). coords(fra1gt_1,277,20,13).  visible(fra1gt_1,13). happensAt(walking(fra1gt_1),13). orientation(fra1gt_7,136,13). coords(fra1gt_7,141,42,13).  visible(fra1gt_7,13). happensAt(walking(fra1gt_7),13).
orientation(fra1gt_1,112,14). coords(fra1gt_1,277,20,14).  visible(fra1gt_1,14). happensAt(walking(fra1gt_1),14). orientation(fra1gt_7,136,14). coords(fra1gt_7,141,42,14).  visible(fra1gt_7,14). happensAt(walking(fra1gt_7),14).
orientation(fra1gt_1,112,15). coords(fra1gt_1,277,20,15).  visible(fra1gt_1,15). happensAt(walking(fra1gt_1),15). orientation(fra1gt_7,136,15). coords(fra1gt_7,141,42,15).  visible(fra1gt_7,15). happensAt(walking(fra1gt_7),15).
orientation(fra1gt_1,112,16). coords(fra1gt_1,277,20,16).  visible(fra1gt_1,16). happensAt(walking(fra1gt_1),16). orientation(fra1gt_7,136,16). coords(fra1gt_7,141,42,16).  visible(fra1gt_7,16). happensAt(walking(fra1gt_7),16).
orientation(fra1gt_1,112,17). coords(fra1gt_1,277,20,17).  visible(fra1gt_1,17). happensAt(walking(fra1gt_1),17). orientation(fra1gt_7,136,17). coords(fra1gt_7,142,42,17).  visible(fra1gt_7,17). happensAt(walking(fra1gt_7),17).
orientation(fra1gt_1,112,18). coords(fra1gt_1,277,19,18).  visible(fra1gt_1,18). happensAt(walking(fra1gt_1),18). orientation(fra1gt_7,136,18). coords(fra1gt_7,142,42,18).  visible(fra1gt_7,18). happensAt(walking(fra1gt_7),18).
orientation(fra1gt_1,112,19). coords(fra1gt_1,277,19,19).  visible(fra1gt_1,19). happensAt(walking(fra1gt_1),19). orientation(fra1gt_7,136,19). coords(fra1gt_7,142,42,19).  visible(fra1gt_7,19). happensAt(walking(fra1gt_7),19).
orientation(fra1gt_1,112,20). coords(fra1gt_1,277,18,20).  visible(fra1gt_1,20). happensAt(walking(fra1gt_1),20). orientation(fra1gt_7,136,20). coords(fra1gt_7,142,42,20).  visible(fra1gt_7,20). happensAt(walking(fra1gt_7),20).
orientation(fra1gt_1,112,21). coords(fra1gt_1,277,16,21).  visible(fra1gt_1,21). happensAt(walking(fra1gt_1),21). orientation(fra1gt_7,136,21). coords(fra1gt_7,142,42,21).  visible(fra1gt_7,21). happensAt(walking(fra1gt_7),21).
orientation(fra1gt_1,108,22). coords(fra1gt_1,277,16,22).  visible(fra1gt_1,22). happensAt(walking(fra1gt_1),22). orientation(fra1gt_7,136,22). coords(fra1gt_7,142,42,22).  visible(fra1gt_7,22). happensAt(walking(fra1gt_7),22).
orientation(fra1gt_1,106,23). coords(fra1gt_1,277,14,23).  visible(fra1gt_1,23). happensAt(walking(fra1gt_1),23). orientation(fra1gt_7,136,23). coords(fra1gt_7,143,41,23).  visible(fra1gt_7,23). happensAt(walking(fra1gt_7),23).
orientation(fra1gt_1,106,24). coords(fra1gt_1,278,12,24).  visible(fra1gt_1,24). happensAt(walking(fra1gt_1),24). orientation(fra1gt_7,136,24). coords(fra1gt_7,144,40,24).  visible(fra1gt_7,24). happensAt(walking(fra1gt_7),24).
orientation(fra1gt_1,106,25). coords(fra1gt_1,278,12,25).  visible(fra1gt_1,25). happensAt(walking(fra1gt_1),25). orientation(fra1gt_7,136,25). coords(fra1gt_7,144,40,25).  visible(fra1gt_7,25). happensAt(walking(fra1gt_7),25).
orientation(fra1gt_1,106,26). coords(fra1gt_1,278,11,26).  visible(fra1gt_1,26). happensAt(walking(fra1gt_1),26). orientation(fra1gt_7,136,26). coords(fra1gt_7,144,40,26).  visible(fra1gt_7,26). happensAt(walking(fra1gt_7),26).
orientation(fra1gt_1,106,27). coords(fra1gt_1,278,11,27).  visible(fra1gt_1,27). happensAt(walking(fra1gt_1),27). orientation(fra1gt_7,136,27). coords(fra1gt_7,144,40,27).  visible(fra1gt_7,27). happensAt(walking(fra1gt_7),27).
orientation(fra1gt_1,106,28). coords(fra1gt_1,278,11,28).  visible(fra1gt_1,28). happensAt(walking(fra1gt_1),28). orientation(fra1gt_7,136,28). coords(fra1gt_7,144,40,28).  visible(fra1gt_7,28). happensAt(walking(fra1gt_7),28).
orientation(fra1gt_1,106,29). coords(fra1gt_1,278,11,29).  visible(fra1gt_1,29). happensAt(walking(fra1gt_1),29). orientation(fra1gt_7,136,29). coords(fra1gt_7,144,40,29).  visible(fra1gt_7,29). happensAt(walking(fra1gt_7),29).
orientation(fra1gt_1,106,30). coords(fra1gt_1,278,11,30).  visible(fra1gt_1,30). happensAt(walking(fra1gt_1),30). orientation(fra1gt_7,136,30). coords(fra1gt_7,144,40,30).  visible(fra1gt_7,30). happensAt(walking(fra1gt_7),30).
orientation(fra1gt_1,106,31). coords(fra1gt_1,278,11,31).  visible(fra1gt_1,31). happensAt(walking(fra1gt_1),31). orientation(fra1gt_7,136,31). coords(fra1gt_7,144,40,31).  visible(fra1gt_7,31). happensAt(walking(fra1gt_7),31).
orientation(fra1gt_1,103,32). coords(fra1gt_1,278,11,32).  visible(fra1gt_1,32). happensAt(walking(fra1gt_1),32). orientation(fra1gt_7,136,32). coords(fra1gt_7,144,40,32).  visible(fra1gt_7,32). happensAt(walking(fra1gt_7),32).
orientation(fra1gt_1,103,33). coords(fra1gt_1,279,10,33).  visible(fra1gt_1,33). happensAt(walking(fra1gt_1),33). orientation(fra1gt_7,136,33). coords(fra1gt_7,144,40,33).  visible(fra1gt_7,33). happensAt(walking(fra1gt_7),33).
orientation(fra1gt_1,103,34). coords(fra1gt_1,279,8,34).  visible(fra1gt_1,34). happensAt(walking(fra1gt_1),34). orientation(fra1gt_7,136,34). coords(fra1gt_7,144,40,34).  visible(fra1gt_7,34). happensAt(walking(fra1gt_7),34).
orientation(fra1gt_1,103,35). coords(fra1gt_1,279,6,35).  visible(fra1gt_1,35). happensAt(walking(fra1gt_1),35). orientation(fra1gt_7,136,35). coords(fra1gt_7,145,40,35).  visible(fra1gt_7,35). happensAt(walking(fra1gt_7),35).
orientation(fra1gt_1,103,36). coords(fra1gt_1,281,6,36).  visible(fra1gt_1,36). happensAt(walking(fra1gt_1),36). orientation(fra1gt_7,136,36). coords(fra1gt_7,145,40,36).  visible(fra1gt_7,36). happensAt(walking(fra1gt_7),36).
#show holdsAt/2.