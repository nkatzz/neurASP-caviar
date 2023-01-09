
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

	orientation(lb1gt_0,155,12). coords(lb1gt_0,175,172,12).  visible(lb1gt_0,12). happensAt(walking(lb1gt_0),12). orientation(lb1gt_1,147,12). coords(lb1gt_1,71,79,12).  visible(lb1gt_1,12). happensAt(walking(lb1gt_1),12).
orientation(lb1gt_0,155,13). coords(lb1gt_0,174,168,13).  visible(lb1gt_0,13). happensAt(walking(lb1gt_0),13). orientation(lb1gt_1,147,13). coords(lb1gt_1,71,79,13).  visible(lb1gt_1,13). happensAt(walking(lb1gt_1),13).
orientation(lb1gt_0,155,14). coords(lb1gt_0,174,167,14).  visible(lb1gt_0,14). happensAt(walking(lb1gt_0),14). orientation(lb1gt_1,147,14). coords(lb1gt_1,71,79,14).  visible(lb1gt_1,14). happensAt(walking(lb1gt_1),14).
orientation(lb1gt_0,155,15). coords(lb1gt_0,174,165,15).  visible(lb1gt_0,15). happensAt(walking(lb1gt_0),15). orientation(lb1gt_1,147,15). coords(lb1gt_1,71,81,15).  visible(lb1gt_1,15). happensAt(walking(lb1gt_1),15).
orientation(lb1gt_0,150,16). coords(lb1gt_0,174,164,16).  visible(lb1gt_0,16). happensAt(walking(lb1gt_0),16). orientation(lb1gt_1,147,16). coords(lb1gt_1,71,81,16).  visible(lb1gt_1,16). happensAt(walking(lb1gt_1),16).
orientation(lb1gt_0,146,17). coords(lb1gt_0,174,164,17).  visible(lb1gt_0,17). happensAt(walking(lb1gt_0),17). orientation(lb1gt_1,147,17). coords(lb1gt_1,71,83,17).  visible(lb1gt_1,17). happensAt(walking(lb1gt_1),17).
orientation(lb1gt_0,144,18). coords(lb1gt_0,173,161,18).  visible(lb1gt_0,18). happensAt(walking(lb1gt_0),18). orientation(lb1gt_1,147,18). coords(lb1gt_1,71,83,18).  visible(lb1gt_1,18). happensAt(walking(lb1gt_1),18).
orientation(lb1gt_0,146,19). coords(lb1gt_0,173,160,19).  visible(lb1gt_0,19). happensAt(walking(lb1gt_0),19). orientation(lb1gt_1,147,19). coords(lb1gt_1,71,84,19).  visible(lb1gt_1,19). happensAt(walking(lb1gt_1),19).
orientation(lb1gt_0,146,20). coords(lb1gt_0,172,156,20).  visible(lb1gt_0,20). happensAt(walking(lb1gt_0),20). orientation(lb1gt_1,147,20). coords(lb1gt_1,71,85,20).  visible(lb1gt_1,20). happensAt(walking(lb1gt_1),20).
orientation(lb1gt_0,146,21). coords(lb1gt_0,171,153,21).  visible(lb1gt_0,21). happensAt(walking(lb1gt_0),21). orientation(lb1gt_1,147,21). coords(lb1gt_1,71,85,21).  visible(lb1gt_1,21). happensAt(walking(lb1gt_1),21).
orientation(lb1gt_0,146,22). coords(lb1gt_0,171,150,22).  visible(lb1gt_0,22). happensAt(walking(lb1gt_0),22). orientation(lb1gt_1,147,22). coords(lb1gt_1,71,86,22).  visible(lb1gt_1,22). happensAt(walking(lb1gt_1),22).
orientation(lb1gt_0,146,23). coords(lb1gt_0,171,149,23).  visible(lb1gt_0,23). happensAt(walking(lb1gt_0),23). orientation(lb1gt_1,147,23). coords(lb1gt_1,71,86,23).  visible(lb1gt_1,23). happensAt(walking(lb1gt_1),23).
orientation(lb1gt_0,146,24). coords(lb1gt_0,171,148,24).  visible(lb1gt_0,24). happensAt(walking(lb1gt_0),24). orientation(lb1gt_1,147,24). coords(lb1gt_1,71,86,24).  visible(lb1gt_1,24). happensAt(walking(lb1gt_1),24).
orientation(lb1gt_0,146,25). coords(lb1gt_0,171,147,25).  visible(lb1gt_0,25). happensAt(walking(lb1gt_0),25). orientation(lb1gt_1,147,25). coords(lb1gt_1,71,86,25).  visible(lb1gt_1,25). happensAt(walking(lb1gt_1),25).
orientation(lb1gt_0,146,26). coords(lb1gt_0,172,146,26).  visible(lb1gt_0,26). happensAt(walking(lb1gt_0),26). orientation(lb1gt_1,147,26). coords(lb1gt_1,71,87,26).  visible(lb1gt_1,26). happensAt(walking(lb1gt_1),26).
orientation(lb1gt_0,146,27). coords(lb1gt_0,172,144,27).  visible(lb1gt_0,27). happensAt(walking(lb1gt_0),27). orientation(lb1gt_1,147,27). coords(lb1gt_1,71,87,27).  visible(lb1gt_1,27). happensAt(walking(lb1gt_1),27).
orientation(lb1gt_0,146,28). coords(lb1gt_0,173,143,28).  visible(lb1gt_0,28). happensAt(walking(lb1gt_0),28). orientation(lb1gt_1,147,28). coords(lb1gt_1,71,87,28).  visible(lb1gt_1,28). happensAt(walking(lb1gt_1),28).
orientation(lb1gt_0,146,29). coords(lb1gt_0,173,141,29).  visible(lb1gt_0,29). happensAt(walking(lb1gt_0),29). orientation(lb1gt_1,147,29). coords(lb1gt_1,71,87,29).  visible(lb1gt_1,29). happensAt(walking(lb1gt_1),29).
orientation(lb1gt_0,146,30). coords(lb1gt_0,174,138,30).  visible(lb1gt_0,30). happensAt(walking(lb1gt_0),30). orientation(lb1gt_1,147,30). coords(lb1gt_1,71,88,30).  visible(lb1gt_1,30). happensAt(walking(lb1gt_1),30).
orientation(lb1gt_0,146,31). coords(lb1gt_0,174,137,31).  visible(lb1gt_0,31). happensAt(walking(lb1gt_0),31). orientation(lb1gt_1,147,31). coords(lb1gt_1,71,88,31).  visible(lb1gt_1,31). happensAt(walking(lb1gt_1),31).
orientation(lb1gt_0,146,32). coords(lb1gt_0,175,134,32).  visible(lb1gt_0,32). happensAt(walking(lb1gt_0),32). orientation(lb1gt_1,147,32). coords(lb1gt_1,72,88,32).  visible(lb1gt_1,32). happensAt(walking(lb1gt_1),32).
orientation(lb1gt_0,149,33). coords(lb1gt_0,175,132,33).  visible(lb1gt_0,33). happensAt(walking(lb1gt_0),33). orientation(lb1gt_1,147,33). coords(lb1gt_1,72,88,33).  visible(lb1gt_1,33). happensAt(walking(lb1gt_1),33).
orientation(lb1gt_0,157,34). coords(lb1gt_0,175,130,34).  visible(lb1gt_0,34). happensAt(walking(lb1gt_0),34). orientation(lb1gt_1,147,34). coords(lb1gt_1,72,88,34).  visible(lb1gt_1,34). happensAt(walking(lb1gt_1),34).
orientation(lb1gt_0,155,35). coords(lb1gt_0,177,129,35).  visible(lb1gt_0,35). happensAt(walking(lb1gt_0),35). orientation(lb1gt_1,147,35). coords(lb1gt_1,72,88,35).  visible(lb1gt_1,35). happensAt(walking(lb1gt_1),35).
orientation(lb1gt_0,156,36). coords(lb1gt_0,177,129,36).  visible(lb1gt_0,36). happensAt(walking(lb1gt_0),36). orientation(lb1gt_1,147,36). coords(lb1gt_1,72,88,36).  visible(lb1gt_1,36). happensAt(walking(lb1gt_1),36).
#show holdsAt/2.